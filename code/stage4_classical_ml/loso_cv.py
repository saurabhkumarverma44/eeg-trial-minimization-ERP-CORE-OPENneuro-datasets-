class LOSOCV:
    """Leave-One-Subject-Out split generator keyed on metadata['subject']."""

    def split(self, metadata):
        subjects = sorted(metadata["subject"].unique())
        for test_subject in subjects:
            yield test_subject