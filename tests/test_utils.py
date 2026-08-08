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
        t_out, x_out = from_feather(path)
        np.testing.assert_allclose(t_out, theta, rtol=1e-5)
        np.testing.assert_allclose(x_out, x, rtol=1e-5)

    def test_nuisance_filter_applied_on_read(self, tmp_path):
        theta = np.ones((10, 3), dtype=np.float32)
        x = np.ones((10, 5), dtype=np.float32)
        path = tmp_path / "nuisance.feather"
        to_feather(path, theta, x)
        nuis_fil = np.ones(3, dtype=bool)
        nuis_fil[-1] = False
        t, _ = from_feather(path, nuisance_filter=nuis_fil)
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
            from_feather(Path("/no/such/file.feather"))

    def test_round_trip_with_many_columns(self, tmp_path):
        """Regression test for the wide-column (~1000 col) read path:
        from_feather must decode via the flatten/reshape route correctly,
        not just for the small column counts most other tests use."""
        rng = np.random.default_rng(0)
        theta = rng.random((37, 5)).astype(np.float32)
        x = rng.random((37, 250)).astype(np.float32)
        path = tmp_path / "wide.feather"
        to_feather(path, theta, x)

        theta_out, x_out = from_feather(path)
        assert x_out.shape == x.shape
        np.testing.assert_allclose(theta_out, theta, rtol=1e-5)
        np.testing.assert_allclose(x_out, x, rtol=1e-5)


class TestCountAndChunkedRead:
    @pytest.fixture()
    def wide_feather_file(self, tmp_path):
        rng = np.random.default_rng(1)
        theta = rng.random((250, 5)).astype(np.float32)
        x = rng.random((250, 100)).astype(np.float32)
        path = tmp_path / "wide.feather"
        to_feather(path, theta, x)
        return path, theta, x

    def test_count_feather_rows_matches_actual_row_count(self, wide_feather_file):
        from mach3sbitools.utils.file_utils import count_feather_rows

        path, theta, _ = wide_feather_file
        assert count_feather_rows(path) == theta.shape[0]

    def test_count_feather_rows_missing_file_raises(self):
        from mach3sbitools.utils.file_utils import count_feather_rows

        with pytest.raises(FileNotFoundError):
            count_feather_rows(Path("/no/such/file.feather"))

    def test_iter_feather_chunks_reassembles_to_original(self, wide_feather_file):
        from mach3sbitools.utils.file_utils import iter_feather_chunks

        path, theta, x = wide_feather_file
        chunks = list(iter_feather_chunks(path, chunk_rows=37))

        # uneven division: 250 / 37 -> 6 full chunks of 37 + 1 of 28
        assert [c[0].shape[0] for c in chunks] == [37] * 6 + [28]

        theta_re = np.concatenate([c[0] for c in chunks])
        x_re = np.concatenate([c[1] for c in chunks])
        np.testing.assert_allclose(theta_re, theta, rtol=1e-5)
        np.testing.assert_allclose(x_re, x, rtol=1e-5)

    def test_iter_feather_chunks_missing_file_raises(self):
        from mach3sbitools.utils.file_utils import iter_feather_chunks

        with pytest.raises(FileNotFoundError):
            list(iter_feather_chunks(Path("/no/such/file.feather"), chunk_rows=10))


class TestListColumnToNumpy:
    """
    Regression coverage for the flatten/reshape fast path that replaced
    `.to_list()`/`.to_pylist()` in from_feather / iter_feather_chunks.

    The critical case here is a *sliced* Arrow list array: `.values` on a
    sliced ListArray returns the full underlying child buffer, ignoring the
    slice, while `.flatten()` correctly respects it — using the wrong one
    silently reads the wrong rows. This is exactly the code path exercised
    by Table.slice() inside iter_feather_chunks.
    """

    def test_matches_naive_to_pylist_decoding(self, tmp_path):
        import pyarrow as pa
        from pyarrow import feather

        from mach3sbitools.utils.file_utils import _list_column_to_numpy

        rng = np.random.default_rng(2)
        x = rng.random((80, 30))
        table = pa.Table.from_pydict({"x": x.tolist()})
        path = tmp_path / "col.feather"
        feather.write_feather(table, str(path))

        tbl = feather.read_table(str(path), memory_map=True)
        fast = _list_column_to_numpy(tbl["x"], dtype=np.float64)
        naive = np.array(tbl["x"].to_pylist(), dtype=np.float64)
        np.testing.assert_allclose(fast, naive)

    def test_sliced_array_reads_correct_rows(self, tmp_path):
        """The regression case: a Table.slice() must not silently return
        values from the full (unsliced) buffer."""
        import pyarrow as pa
        from pyarrow import feather

        from mach3sbitools.utils.file_utils import _list_column_to_numpy

        rng = np.random.default_rng(3)
        x = rng.random((200, 12))
        table = pa.Table.from_pydict({"x": x.tolist()})
        path = tmp_path / "col.feather"
        feather.write_feather(table, str(path))

        tbl = feather.read_table(str(path), memory_map=True)
        sliced_tbl = tbl.slice(57, 23)
        sliced = _list_column_to_numpy(sliced_tbl["x"], dtype=np.float64)

        np.testing.assert_allclose(sliced, x[57 : 57 + 23])
