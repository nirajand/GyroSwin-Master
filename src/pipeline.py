import lightning as L
import zarr
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Tuple

class GyroDataset(Dataset):
    """
    Memory-efficient Zarr dataset with lazy-loading and worker-safe handles.
    """
    def __init__(self, path: str):
        self.path = path
        # Open initially just to get metadata/length
        store = zarr.DirectoryStore(path)
        root = zarr.open(store, mode='r')
        self.length = root.f.shape[0]
        self.data = None  # Handle will be initialized per-worker

    def _init_db(self):
        """Initializes the Zarr handle for the current process/worker."""
        if self.data is None:
            store = zarr.DirectoryStore(self.path)
            self.data = zarr.open(store, mode='r')

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._init_db()
        
        # Load slices from Zarr (returning numpy)
        f = self.data.f[i]     # [T, X, Y, V, Mu]
        phi = self.data.phi[i] # [T, X, Y]
        
        # Convert to tensor and cast to float32
        f_tensor = torch.from_numpy(f).float()
        phi_tensor = torch.from_numpy(phi).float()
        
        # In production, we usually shift f to t+1 for the target
        # For this multi-task example, we'll return:
        # Input (f_t), Target_Next_Step (f_t+1), Target_Potential (phi_t)
        return f_tensor, f_tensor, phi_tensor

class GyroDataModule(L.LightningDataModule):
    def __init__(
        self, 
        path: str, 
        batch_size: int = 32, 
        num_workers: int = 4,
        train_val_split: float = 0.8
    ):
        super().__init__()
        self.save_hyperparameters()
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = train_val_split

    def setup(self, stage: Optional[str] = None):
        full_dataset = GyroDataset(self.path)
        
        # Deterministic splitting for reproducibility
        train_size = int(self.split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        self.train_ds, self.val_ds = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,  # Crucial for fast GPU transfer
            persistent_workers=True, # Keeps Zarr handles alive between epochs
            prefetch_factor=2
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
