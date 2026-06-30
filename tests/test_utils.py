"""
Tests for mach3sbitools.utils — device_handler and file_utils.

Logger tests are omitted: they're thin wrappers over stdlib logging and Rich,
so there's no meaningful behaviour to assert beyond "it doesn't raise".
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mach3sbitools.utils.device_handler import TensorConversionError, TorchDeviceHandler
from mach3sbitools.utils.file_utils import from_feather, to_feather

# ─────────────────────────────────────────────────────────────────────────────
# TorchDeviceHandler
# ─────────────────────────────────────────────────────────────────────────────


class TestTorchDeviceHandler:
    def test_device_is_valid(self):
        assert TorchDeviceHandler().device in ("cpu", "cuda")

    def test_to_tensor_from_ndarray(self):
        t = TorchDeviceHandler().to_tensor(np.array([1.0, 2.0], dtype=np.float32))
        assert isinstance(t, torch.Tensor)
        assert t.shape == (2,)

    def test_to_tensor_from_dataframe(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        t = TorchDeviceHandler().to_tensor(df)
        assert t.shape == (2, 2)

    def test_to_tensor_raises_on_unconvertible(self):
        with pytest.raises(TensorConversionError):
            TorchDeviceHandler().to_tensor(object())


# ─────────────────────────────────────────────────────────────────────────────
# Feather I/O
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatherIO:
    @pytest.fixture()
    def feather_file(self, tmp_path):
        theta = np.random.rand(20, 4).astype(np.float32)
        x = np.random.rand(20, 6).astype(np.float32)
        path = tmp_path / "data.feather"
        to_feather(path, theta, x)
        return path, theta, x

    def test_round_trip_preserves_values(self, feather_file):
        path, theta, x = feather_file
        t_out, x_out = from_feather(path, [f"p{i}" for i in range(4)])
        np.testing.assert_allclose(t_out, theta, rtol=1e-5)
        np.testing.assert_allclose(x_out, x, rtol=1e-5)

    def test_nuisance_filter_applied_on_read(self, tmp_path):
        theta = np.ones((10, 3), dtype=np.float32)
        x = np.ones((10, 5), dtype=np.float32)
        path = tmp_path / "nuisance.feather"
        to_feather(path, theta, x)
        t, _ = from_feather(path, ["keep", "drop_x", "keep2"], nuisance_pars=["drop_*"])
        assert t.shape == (10, 2)

    def test_raises_on_wrong_suffix(self, tmp_path):
        with pytest.raises(ValueError, match="feather"):
            to_feather(
                tmp_path / "out.csv",
                np.ones((5, 2), dtype=np.float32),
                np.ones((5, 3), dtype=np.float32),
            )

    def test_raises_if_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            from_feather(Path("/no/such/file.feather"), ["a"])
