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

from mach3sbitools.utils import get_device, reset_device_cache, to_tensor
from mach3sbitools.utils.device_handler import DEVICE_ENV_VAR, TensorConversionError
from mach3sbitools.utils.feather_utils import from_feather, peek_num_rows, to_feather

# ─────────────────────────────────────────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_device_cache():
    """Keep the cached device from leaking between tests."""
    reset_device_cache()
    yield
    reset_device_cache()


class TestGetDevice:
    def test_returns_a_torch_device(self):
        """A torch.device, not a string — so `tensor.device == get_device()` works."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cpu", "cuda")

    def test_compares_equal_to_a_tensor_device(self):
        assert torch.zeros(1, device=get_device()).device == get_device()

    def test_is_cached(self):
        assert get_device() is get_device()

    def test_environment_override_is_respected(self, monkeypatch):
        monkeypatch.setenv(DEVICE_ENV_VAR, "cpu")
        reset_device_cache()
        assert get_device() == torch.device("cpu")

    def test_reset_picks_up_a_changed_environment(self, monkeypatch):
        monkeypatch.setenv(DEVICE_ENV_VAR, "cpu")
        reset_device_cache()
        assert get_device().type == "cpu"
        monkeypatch.delenv(DEVICE_ENV_VAR)
        reset_device_cache()
        assert get_device().type in ("cpu", "cuda")

    def test_mps_is_not_selected_automatically(self, monkeypatch):
        """MPS has no float64, and the priors are evaluated in double."""
        monkeypatch.delenv(DEVICE_ENV_VAR, raising=False)
        reset_device_cache()
        assert get_device().type != "mps"


class TestToTensor:
    def test_from_ndarray(self):
        t = to_tensor(np.array([1.0, 2.0], dtype=np.float32))
        assert isinstance(t, torch.Tensor)
        assert t.shape == (2,)

    def test_from_dataframe(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        assert to_tensor(df).shape == (2, 2)

    def test_from_list(self):
        assert to_tensor([1.0, 2.0, 3.0]).shape == (3,)

    def test_from_tensor_is_detached_copy(self):
        source = torch.ones(3, requires_grad=True)
        out = to_tensor(source)
        assert not out.requires_grad
        assert out.data_ptr() != source.data_ptr()

    def test_lands_on_the_default_device(self):
        assert to_tensor([1.0]).device == get_device()

    def test_explicit_device_overrides_the_default(self):
        assert to_tensor([1.0], device="cpu").device == torch.device("cpu")

    def test_raises_on_unconvertible(self):
        with pytest.raises(TensorConversionError):
            to_tensor(object())


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
        t_out, x_out = from_feather(path)
        np.testing.assert_allclose(t_out, theta, rtol=1e-5)
        np.testing.assert_allclose(x_out, x, rtol=1e-5)

    def test_round_trip_preserves_shape(self, feather_file):
        path, theta, x = feather_file
        t_out, x_out = from_feather(path)
        assert t_out.shape == theta.shape
        assert x_out.shape == x.shape

    def test_peek_num_rows_matches_contents(self, feather_file):
        path, theta, _ = feather_file
        assert peek_num_rows(path) == theta.shape[0]

    def test_raises_on_wrong_suffix(self, tmp_path):
        with pytest.raises(ValueError, match="feather"):
            to_feather(
                tmp_path / "out.csv",
                np.ones((5, 2), dtype=np.float32),
                np.ones((5, 3), dtype=np.float32),
            )

    def test_raises_if_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            from_feather(Path("/no/such/file.feather"))
