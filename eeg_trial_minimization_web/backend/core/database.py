import aiosqlite
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any
from backend.core.config import settings

logger = logging.getLogger(__name__)

@asynccontextmanager
async def get_db_connection():
    settings.EEG_DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(str(settings.EEG_DATABASE_PATH)) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON;")
        yield db

async def init_database():
    """Initialize database tables and register built-in research checkpoints."""
    settings.EEG_DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with get_db_connection() as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                format TEXT NOT NULL,
                architecture TEXT NOT NULL,
                num_channels INTEGER NOT NULL,
                sampling_rate REAL DEFAULT 1000.0,
                input_shape TEXT,
                class_labels TEXT,
                preprocessing_info TEXT,
                is_active INTEGER DEFAULT 0,
                is_builtin INTEGER DEFAULT 0,
                validation_status TEXT DEFAULT 'pending',
                validation_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                task_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                model_id TEXT,
                trial_fractions TEXT,
                trial_counts TEXT,
                num_folds INTEGER DEFAULT 5,
                random_seed INTEGER DEFAULT 42,
                cv_strategy TEXT DEFAULT 'loso',
                augmentation_config TEXT,
                status TEXT DEFAULT 'created',
                current_step TEXT,
                progress REAL DEFAULT 0.0,
                summary_metrics TEXT,
                error_message TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS experiment_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                trial_fraction REAL NOT NULL,
                n_train_trials INTEGER NOT NULL,
                fold INTEGER,
                test_acc REAL NOT NULL,
                balanced_acc REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                roc_auc REAL,
                confusion_matrix TEXT,
                metrics_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE
            );
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                model_name TEXT,
                source_type TEXT,
                epoch_index INTEGER,
                predicted_class INTEGER,
                predicted_label TEXT,
                class_probabilities TEXT,
                inference_time_ms REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        await db.commit()

        # Seed built-in EEGNet research checkpoint if present
        builtin_eegnet = settings.EEG_PROJECT_ROOT / "results_dl_1" / "checkpoints_eegnet" / "eegnet_DS002680_gonogo_4_best.pt"
        if builtin_eegnet.exists():
            cursor = await db.execute("SELECT id FROM models WHERE id = 'eegnet-ds002680-gonogo-4'")
            row = await cursor.fetchone()
            if not row:
                await db.execute("""
                    INSERT INTO models (
                        id, name, filename, filepath, format, architecture,
                        num_channels, sampling_rate, input_shape, class_labels,
                        preprocessing_info, is_active, is_builtin, validation_status, validation_notes
                    ) VALUES (
                        'eegnet-ds002680-gonogo-4',
                        'EEGNet DS002680 Go/No-Go #4 (Verified Research Model)',
                        'eegnet_DS002680_gonogo_4_best.pt',
                        ?,
                        '.pt',
                        'EEGNet',
                        31,
                        1000.0,
                        '[1, 31, 1001]',
                        '["Go (0)", "No-Go (1)"]',
                        '{"bandpass": [0.1, 40.0], "epoch_tmin": -0.2, "epoch_tmax": 0.8, "baseline": [-0.2, 0.0]}',
                        1,
                        1,
                        'valid',
                        'Pre-trained best checkpoint on DS002680 Go/No-Go task 4. 95.93% test accuracy.'
                    );
                """, (str(builtin_eegnet),))
                await db.commit()
                logger.info("Registered built-in research EEGNet checkpoint.")

        # Seed 8-Channel Neuro Cap model checkpoint
        cap_8ch_model = Path("storage/models/eegnet_8ch_neurocap.pt")
        if cap_8ch_model.exists():
            cursor = await db.execute("SELECT id FROM models WHERE id = 'eegnet-8ch-neurocap'")
            row = await cursor.fetchone()
            if not row:
                await db.execute("""
                    INSERT INTO models (
                        id, name, filename, filepath, format, architecture,
                        num_channels, sampling_rate, input_shape, class_labels,
                        preprocessing_info, is_active, is_builtin, validation_status, validation_notes
                    ) VALUES (
                        'eegnet-8ch-neurocap',
                        'EEGNet 8-Channel Neuro Cap (Fp1, Fp2, C3, C4, P3, P4, O1, O2)',
                        'eegnet_8ch_neurocap.pt',
                        ?,
                        '.pt',
                        'EEGNet',
                        8,
                        250.0,
                        '[1, 8, 1001]',
                        '["Go (0)", "No-Go (1)"]',
                        '{"bandpass": [0.5, 35.0], "epoch_tmin": -0.2, "epoch_tmax": 0.8, "baseline": [-0.2, 0.0]}',
                        0,
                        0,
                        'valid',
                        'Calibrated model for 8-channel physical neuro cap. Optimized for Go/No-Go trial minimisation.'
                    );

                """, (str(cap_8ch_model.resolve()),))
                await db.commit()
                logger.info("Registered 8-Channel Neuro Cap EEGNet model.")

