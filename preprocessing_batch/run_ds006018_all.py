# preprocessing_batch/run_ds006018_all.py

from ds006018_pipeline import run_ds006018_preprocessing


def main():
    run_ds006018_preprocessing(overwrite=False)


if __name__ == "__main__":
    main()