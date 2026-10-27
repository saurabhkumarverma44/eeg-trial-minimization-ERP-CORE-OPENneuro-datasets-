"""
models.py - Stage 4 Classical ML model wrappers
WBS 4.2: RF  n_estimators=200, max_depth=10
WBS 4.3: XGB lr=0.1, max_depth=6, subsample=0.8, A100 GPU
WBS 4.4: SVM RBF kernel with inner CV for C in [0.1,1,10]
         and gamma in [scale, 0.01]. 2-fold inner CV.
         Tighter grid chosen for computational feasibility
         while maintaining WBS inner-CV requirement.
All models implement: fit(), predict(), predict_proba(),
                      train_predict(), feature_importance()
NOTE: feature_importance() is a CALLABLE METHOD (not property).
"""
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

USE_GPU = os.environ.get("STAGE4_USE_GPU", "1") == "1"

try:
    import packaging.version as _pv
    _XGB_GE_2 = _pv.parse(xgb.__version__) >= _pv.parse("2.0.0")
except Exception:
    _XGB_GE_2 = False

try:
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuRF
    _CUML_AVAILABLE = True
except ImportError:
    _CUML_AVAILABLE = False


class RFModel:
    """WBS 4.2 RF: n_estimators=200, max_depth=10."""
    def __init__(self, n_estimators=200, max_depth=10, **kwargs):
        self._importances = None
        if USE_GPU and _CUML_AVAILABLE:
            self.clf = cuRF(n_estimators=n_estimators,
                            max_depth=max_depth,
                            n_streams=4, random_state=42)
        else:
            self.clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=5,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=42)

    def fit(self, X, y):
        self.clf.fit(X, y)
        self._importances = np.asarray(self.clf.feature_importances_)
        return self

    def predict(self, X):
        p = self.clf.predict(X)
        return (p.to_numpy() if hasattr(p, "to_numpy") else np.asarray(p)).astype(int)

    def predict_proba(self, X):
        p = self.clf.predict_proba(X)
        return p.to_numpy() if hasattr(p, "to_numpy") else np.asarray(p)

    def train_predict(self, X_train, y_train, X_test):
        self.fit(X_train, y_train)
        return self.predict(X_test), self.predict_proba(X_test)

    def feature_importance(self):
        return self._importances


class SVMModel:
    """
    WBS 4.4 SVM RBF with inner CV for C and gamma.
    Grid: C in [0.1, 1, 10], gamma in [scale, 0.01].
    Inner 2-fold StratifiedKFold CV (computationally feasible).
    best_params_ saved per WBS requirement.
    """
    def __init__(self, **kwargs):
        self._importances = None
        self.best_params_ = {}
        self._param_grid = {
            "C":     [0.1, 1.0, 10.0],
            "gamma": ["scale", 0.01],
        }

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        base = SVC(kernel="rbf", class_weight="balanced",
                   probability=False, random_state=42)
        gs = GridSearchCV(base, self._param_grid, cv=inner_cv,
                          scoring="balanced_accuracy",
                          n_jobs=-1, refit=True)
        gs.fit(X, y)
        self.best_params_ = gs.best_params_
        best_base = SVC(kernel="rbf", class_weight="balanced",
                        probability=False, random_state=42,
                        **gs.best_params_)
        self.clf = CalibratedClassifierCV(best_base, cv=2, ensemble=False)
        self.clf.fit(X, y)
        self._importances = np.zeros(X.shape[1], dtype=np.float32)
        return self

    def predict(self, X):
        return np.asarray(self.clf.predict(X)).astype(int)

    def predict_proba(self, X):
        return np.asarray(self.clf.predict_proba(X))

    def train_predict(self, X_train, y_train, X_test):
        self.fit(X_train, y_train)
        return self.predict(X_test), self.predict_proba(X_test)

    def feature_importance(self):
        """SVM RBF has no native feature importance. Returns zero array."""
        return self._importances


class XGBModel:
    """WBS 4.3 XGBoost: lr=0.1, max_depth=6, subsample=0.8. GPU A100."""
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=6,
                 subsample=0.8, colsample_bytree=0.8, **kwargs):
        self._importances = None
        params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=3,
            reg_alpha=0.05,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1)
        if USE_GPU:
            if _XGB_GE_2:
                params["device"] = "cuda"
            else:
                params["tree_method"] = "gpu_hist"
                params["gpu_id"] = 0
        params.update(kwargs)
        self.clf = xgb.XGBClassifier(**params)

    def fit(self, X, y):
        self.clf.fit(X, y)
        self._importances = np.asarray(self.clf.feature_importances_)
        return self

    def predict(self, X):
        return np.asarray(self.clf.predict(X)).astype(int)

    def predict_proba(self, X):
        return np.asarray(self.clf.predict_proba(X))

    def train_predict(self, X_train, y_train, X_test):
        self.fit(X_train, y_train)
        return self.predict(X_test), self.predict_proba(X_test)

    def feature_importance(self):
        return self._importances


XGBoostModel      = XGBModel
RandomForestModel = RFModel

MODEL_BUILDERS = {
    "random_forest": RFModel,
    "xgboost":       XGBModel,
    "svm":           SVMModel,
}
