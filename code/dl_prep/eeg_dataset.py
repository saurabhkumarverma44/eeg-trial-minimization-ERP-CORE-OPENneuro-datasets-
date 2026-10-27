import torch
from torch.utils.data import Dataset, DataLoader

class EEGDataset(Dataset):
    """
    PyTorch Dataset for EEG trial tensors.
    """
    def __init__(self, X, y, subject_ids=None):
        if isinstance(X, torch.Tensor):
            self.X = X.clone().detach().float()
        else:
            self.X = torch.from_numpy(X).float()

        if isinstance(y, torch.Tensor):
            self.y = y.clone().detach().long()
        else:
            self.y = torch.from_numpy(y).long()

        self.subject_ids = subject_ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_dataloaders(train_data, val_data, test_data, batch_size=32, num_workers=4):
    X_tr, y_tr, sub_tr = train_data
    X_va, y_va, sub_va = val_data
    X_te, y_te, sub_te = test_data

    train_ds = EEGDataset(X_tr, y_tr, sub_tr)
    val_ds = EEGDataset(X_va, y_va, sub_va)
    test_ds = EEGDataset(X_te, y_te, sub_te)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
