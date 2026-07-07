# preprocessing_batch/run_erpcore_all.py

from erpcore_pipeline import run_erpcore_preprocessing


def main():
    # overwrite=False → resume support using DONE flags
    run_erpcore_preprocessing(overwrite=False)


if __name__ == "__main__":
    main()