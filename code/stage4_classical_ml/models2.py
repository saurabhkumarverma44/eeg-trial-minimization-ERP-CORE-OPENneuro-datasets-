"""
models2.py - Stage 4 GPU-optimised model wrappers
RF:  WBS 4.2 params (n_estimators=200, max_depth=10), CPU n_jobs=-1
XGB: WBS 4.3 params, A100 GPU via device=cuda
SVM: WBS 4.4 RBF kernel, NO inner CV (fixed C=1, gamma=scale)
     Fast version for parallel GPU comparison run.
     Results saved to results_stage4_gpu/ (separate from WBS run).
"""
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

USE_GPU = os.environ.get("STAGE4_USE_GPU", "1") == "1"

try:
    import packaging.version as _pv
    _XGB_GE_2 = _pv.parse(xgb.__version__) >= _pv.parse("2.0.0")
except Exception:
    _XGB_GE_2 = False

# Try cuML for GPU RF and SVM
try:
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.svm import SVC as cuSVC
    _CUML_AVAILABLE = True
    print("[models2] cuML available: GPU RF + SVM enabled")
except ImportError:
    _CUML_AVAILABLE = False
    print("[models2] cuML not found: RF=CPU n_jobs=-1, SVM=CPU fast fixed-C")


class RFModel:
    """WBS 4.2 RF: n_estimators=200, max_depth=10. GPU if cuML else CPU."""
    def __init__(self, n_estimators=200, max_depth=10, **kwargs):
        self._importances = None
        if USE_GPU and _CUML_AVAILABLE:
            self.clf = cuRF(n_estimators=n_estimators,
                            max_depth=max_depth,
                            n_streams=4, random_state=42)
            self._gpu = True
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
            self._gpu = False

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
    WBS 4.4 SVM RBF — fast fixed-C version for GPU comparison run.
    C=1, gamma=scale (no inner CV grid search).
    Uses CalibratedClassifierCV for probability output.
    GPU via cuML if available.
    """
    def __init__(self, C=1.0, gamma="scale", **kwargs):
        self._importances = None
        self.best_params_ = {"C": C, "gamma": gamma, "note": "fixed_no_inner_cv"}
        self._C = C
        self._gamma = gamma

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if USE_GPU and _CUML_AVAILABLE:
            from cuml.svm import SVC as cuSVC
            # cuML SVC: fast GPU, no probability — wrap with sklearn calibration
            # Note: cuML predict returns float, cast to int
            self._cuml_clf = cuSVC(C=self._C, kernel="rbf",
                                    gamma=self._gamma if self._gamma != "scale"
                                    else 1.0 / X.shape[1],
                                    probability=True)
            self._cuml_clf.fit(X, y)
            self._use_cuml = True
        else:
            base = SVC(C=self._C, kernel="rbf", gamma=self._gamma,
                       class_weight="balanced",
                       probability=False, random_state=42)
            self.clf = CalibratedClassifierCV(base, cv=2, ensemble=False)
            self.clf.fit(X, y)
            self._use_cuml = False

        self._importances = np.zeros(X.shape[1], dtype=np.float32)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if getattr(self, "_use_cuml", False):
            p = self._cuml_clf.predict(X)
            return (p.to_numpy() if hasattr(p, "to_numpy") else np.asarray(p)).astype(int)
        return np.asarray(self.clf.predict(X)).astype(int)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if getattr(self, "_use_cuml", False):
            p = self._cuml_clf.predict_proba(X)
            return p.to_numpy() if hasattr(p, "to_numpy") else np.asarray(p)
        return np.asarray(self.clf.predict_proba(X))

    def train_predict(self, X_train, y_train, X_test):
        self.fit(X_train, y_train)
        return self.predict(X_test), self.predict_proba(X_test)

    def feature_importance(self):
        return self._importances


class XGBModel:
    """WBS 4.3 XGBoost: lr=0.1, max_depth=6, subsample=0.8. A100 GPU."""
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
