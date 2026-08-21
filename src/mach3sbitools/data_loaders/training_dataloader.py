import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from mach3sbitools.simulator import Prior
from mach3sbitools.utils import get_logger
 
class TrainingDataset(Dataset):
    def __init__(self, theta_path: Path, x_path: Path, prior: Prior):
        self.theta_path = theta_path
        self.x_path = x_path
 
        # Read only the header here to get shape/dtype/length cheaply,
        # without holding an open memmap across a potential fork.
        theta_header = np.load(theta_path, mmap_mode="r")
        self._len = theta_header.shape[0]
        del theta_header
 
        # Actual memmaps are opened lazily per worker process (see below),
        # so nothing large is pickled when DataLoader spawns workers.
        self._theta = None
        self._x = None
        
        self._nuisance_filter = prior.nuisance_filter.cpu().numpy()
 
    def _ensure_open(self):
        if self._theta is None:
            self._theta = np.load(self.theta_path, mmap_mode="r")
            self._x = np.load(self.x_path, mmap_mode="r")
 
    def __len__(self):
        return self._len
 
    def __getitems__(self, indices: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        self._ensure_open()
        assert self._theta is not None and self._x is not None

        idx_arr = np.asarray(indices)
        theta_batch = torch.from_numpy(np.array(self._theta[idx_arr])).float()
        x_batch = torch.from_numpy(np.array(self._x[idx_arr])).float()

        if theta_batch.shape[1] == len(self._nuisance_filter):
            theta_batch = theta_batch[:, self._nuisance_filter]
        else:
            theta_batch = theta_batch[:, :, self._nuisance_filter]  # adjust to your actual shape

        return list(zip(theta_batch, x_batch))
 
    def __getitem__(self, idx):
        self._ensure_open()

        assert self._theta is not None 
        assert self._x is not None
    
        theta = torch.from_numpy(np.array(self._theta[idx])).float()
        x = torch.from_numpy(np.array(self._x[idx])).float()
        
        if theta.shape[0] == len(self._nuisance_filter):
            theta = theta[self._nuisance_filter]
        else:
            theta = theta[:,self._nuisance_filter]
        
        return theta, x