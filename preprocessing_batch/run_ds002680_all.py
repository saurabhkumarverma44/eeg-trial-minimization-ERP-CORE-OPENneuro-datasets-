# preprocessing_batch/run_ds002680_all.py

from ds002680_pipeline import run_ds002680_preprocessing


def main():
    run_ds002680_preprocessing(overwrite=False)


if __name__ == "__main__":
    main()