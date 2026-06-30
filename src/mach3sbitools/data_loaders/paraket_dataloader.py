"""
Dataset implementation for feather-based simulation files.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm

from mach3sbitools.utils import from_feather
from mach3sbitools.simulator import Prior

class ParaketDataset(Dataset):
    """
    File-level PyTorch dataset over a folder of ``.feather`` simulation files.

    Each ``__getitem__`` call loads one feather file and returns a
    ``(theta, x)`` pair. Call :meth:`to_tensor_dataset` before training to
    pre-load everything into RAM as a flat :class:`~torch.utils.data.TensorDataset`.
    """

    def __init__(
        self,
        data_folder: Path,
        prior: Prior
    ):
        """
        :param data_folder: Directory containing ``.feather`` files.
        :param parameter_names: Ordered list of parameter names in each file's
            ``theta`` column.
        :param nuisance_params: fnmatch patterns for parameters to filter out
            of *theta* on load.
        """
        if not isinstance(data_folder, Path):
            data_folder = Path(data_folder)

        self.data_folder = data_folder

        self.files = sorted(data_folder.glob("*.feather"))
        self.prior = prior

    def __len__(self) -> int:
        """Number of feather files in the dataset folder."""
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load one feather file and return ``(theta, x)`` as tensors.

        :param idx: File index.
        :returns: Tuple of ``(theta, x)`` float tensors.
        """
        theta, x = from_feather(
            self.files[idx], self.prior.nuisance_filter
        )
        return torch.from_numpy(theta), torch.from_numpy(x)

    def to_tensor_dataset(
        self, device: str = "cpu", verbose: bool = True
    ) -> TensorDataset:
        """
        Pre-load all feather files into a flat :class:`~torch.utils.data.TensorDataset`.

        Concatenates all ``(theta, x)`` pairs along the sample dimension.
        This avoids repeated disk reads per epoch during training.

        :param device: Target device for the output tensors.
        :returns: A :class:`~torch.utils.data.TensorDataset` of
            ``(theta_tensor, x_tensor)`` with shapes
            ``(n_total_samples, n_params)`` and ``(n_total_samples, n_bins)``.
        """
        all_theta, all_x = [], []

        for idx in tqdm(
            range(len(self)), desc="Pre-loading dataset", disable=not verbose
        ):
            theta, x = self[idx]
            all_theta.append(theta)
            all_x.append(x)

        theta_tensor = torch.cat(all_theta, dim=0).to(device)
        x_tensor = torch.cat(all_x, dim=0).to(device)

        if verbose:
            print(
                f"Loaded {theta_tensor.shape[0]:,} simulations | "
                f"θ: {theta_tensor.shape[1]}D  x: {x_tensor.shape[1]}D | "
                f"RAM: {(theta_tensor.nbytes + x_tensor.nbytes) / 1e9:.2f} GB"
            )

        return TensorDataset(theta_tensor, x_tensor)
