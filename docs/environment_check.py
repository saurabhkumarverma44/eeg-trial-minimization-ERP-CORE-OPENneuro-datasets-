import mne
import sklearn
import torch
import pandas as pd
import numpy as np

print("=" * 50)
print("PHASE 1.1 ENVIRONMENT CHECK")
print("=" * 50)

print("MNE Version       :", mne.__version__)
print("PyTorch Version   :", torch.__version__)
print("Scikit-Learn      :", sklearn.__version__)
print("Pandas Version    :", pd.__version__)
print("NumPy Version     :", np.__version__)

print("\nEnvironment Setup Successful")