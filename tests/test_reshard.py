"""
Tests for mach3sbitools.apps.reshard.
"""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from mach3sbitools.apps import cli
from mach3sbitools.apps.reshard import reshard_module
from mach3sbitools.data_loaders import StreamingFeatherDataset
from mach3sbitools.utils import from_feather, to_feather


@pytest.fixture
def monolithic_file(tmp_path_factory) -> Path:
    """A single large-ish feather file, simulating a dataset pre-generated
    in one `simulate` call before shard-friendly workflows existed."""
    folder: Path = tmp_path_factory.mktemp("monolithic")
    file = folder / "all_sims.feather"
    rng = np.random.default_rng(1)
    theta = rng.random((10007, 6)).astype(np.float64)
    x = rng.random((10007, 5)).astype(np.float64)
    to_feather(file, theta, x)
    return file


def _ground_truth(file: Path) -> tuple[np.ndarray, np.ndarray]:
    return from_feather(file)


class TestReshardModule:
    def test_reshards_monolithic_file_into_correct_shard_count(
        self, monolithic_file, tmp_path
    ):
        out = tmp_path / "sharded"
        reshard_module(monolithic_file, out, rows_per_shard=2000, prefix="shard")

        shards = sorted(out.glob("*.feather"))
        # 10007 rows / 2000 per shard -> 6 shards (5 full + 1 remainder)
        assert len(shards) == 6

    def test_reshard_is_bit_exact_round_trip(self, monolithic_file, tmp_path):
        out = tmp_path / "sharded"
        reshard_module(monolithic_file, out, rows_per_shard=1500, prefix="shard")

        theta_gt, x_gt = _ground_truth(monolithic_file)

        thetas, xs = [], []
        for f in sorted(out.glob("*.feather")):
            th, x = from_feather(f)
            thetas.append(th)
            xs.append(x)
        theta_re = np.concatenate(thetas)
        x_re = np.concatenate(xs)

        assert theta_re.shape == theta_gt.shape
        assert x_re.shape == x_gt.shape
        assert np.allclose(theta_re, theta_gt)
        assert np.allclose(x_re, x_gt)

    def test_reshard_output_consumable_by_streaming_dataset(
        self, monolithic_file, tmp_path, prior
    ):
        out = tmp_path / "sharded"
        reshard_module(monolithic_file, out, rows_per_shard=2000, prefix="shard")

        theta_gt, _ = _ground_truth(monolithic_file)
        ds = StreamingFeatherDataset(out, prior, cache_size=2)
        assert len(ds) == theta_gt.shape[0]

    def test_reshard_folder_of_files(self, tmp_path):
        """Resharding a folder (not a single file) processes every
        .feather file in it independently."""
        folder = tmp_path / "in"
        folder.mkdir()
        rng = np.random.default_rng(2)
        for i in range(3):
            theta = rng.random((500, 4)).astype(np.float64)
            x = rng.random((500, 3)).astype(np.float64)
            to_feather(folder / f"f{i}.feather", theta, x)

        out = tmp_path / "out"
        reshard_module(folder, out, rows_per_shard=300, prefix="shard")

        total_rows = sum(from_feather(f)[0].shape[0] for f in out.glob("*.feather"))
        assert total_rows == 1500

    def test_missing_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            reshard_module(
                tmp_path / "nope.feather",
                tmp_path / "out",
                rows_per_shard=100,
                prefix="s",
            )

    def test_empty_input_folder_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            reshard_module(empty, tmp_path / "out", rows_per_shard=100, prefix="s")


class TestReshardCLI:
    def test_cli_invocation(self, monolithic_file, tmp_path):
        out = tmp_path / "sharded"
        result = CliRunner().invoke(
            cli,
            [
                "reshard",
                "-i",
                str(monolithic_file),
                "-o",
                str(out),
                "--rows_per_shard",
                "2500",
            ],
        )
        assert result.exit_code == 0, result.output
        shards = list(out.glob("*.feather"))
        assert len(shards) == 5  # ceil(10007 / 2500)

    def test_cli_missing_input_errors(self, tmp_path):
        result = CliRunner().invoke(
            cli,
            [
                "reshard",
                "-i",
                str(tmp_path / "does_not_exist.feather"),
                "-o",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0
