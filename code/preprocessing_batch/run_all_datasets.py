# preprocessing_batch/run_all_datasets.py

from erpcore_pipeline import run_erpcore_preprocessing
from ds006018_pipeline import run_ds006018_preprocessing
from ds002680_pipeline import run_ds002680_preprocessing


def main():
    # Order doesn’t matter; datasets are independent.
    run_erpcore_preprocessing(overwrite=False)
    run_ds006018_preprocessing(overwrite=False)
    run_ds002680_preprocessing(overwrite=False)


if __name__ == "__main__":
    main()