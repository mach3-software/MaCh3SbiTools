"""
Tests for mach3sbitools.diagnostics.compare_log.

``compare_logl`` is a plotting routine over two expensive objects, so the
simulator, the handler and matplotlib are mocked and the assertions are about
what it computes and hands to the axes.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mach3sbitools.diagnostics import compare_logl
from mach3sbitools.diagnostics.compare_log import normalise_logl

N_BINS = 12


# ─────────────────────────────────────────────────────────────────────────────
# normalise_logl
# ─────────────────────────────────────────────────────────────────────────────


class TestNormaliseLogl:
    def test_standardises_the_input(self):
        result = normalise_logl(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert result.mean() == pytest.approx(0.0, abs=1e-10)
        assert result.std() == pytest.approx(1.0, abs=1e-10)

    def test_constant_input_returns_zeros(self):
        """A zero standard deviation must not produce nan."""
        result = normalise_logl(np.array([3.0, 3.0, 3.0]))
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_shape_is_preserved(self):
        arr = np.random.rand(50)
        assert normalise_logl(arr).shape == arr.shape

    def test_ordering_is_preserved(self):
        """Standardising is monotonic — the ranking must survive it."""
        arr = np.array([5.0, 1.0, 3.0, 9.0])
        np.testing.assert_array_equal(np.argsort(normalise_logl(arr)), np.argsort(arr))


# ─────────────────────────────────────────────────────────────────────────────
# compare_logl
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def mocks():
    """Minimal Simulator and InferenceHandler stand-ins."""

    def _build(n_samples: int = 20):
        handler = MagicMock()
        handler.sample_posterior.return_value.cpu.return_value.numpy.return_value = (
            np.random.randn(n_samples, 5).astype(np.float32)
        )
        handler.get_log_likelihood.return_value.cpu.return_value.numpy.return_value = (
            np.random.randn(n_samples).astype(np.float64)
        )

        simulator = MagicMock()
        simulator.simulator_wrapper.get_data_bins.return_value = np.ones(
            N_BINS
        ).tolist()
        simulator.simulator_wrapper.get_log_likelihood.return_value = 1.0

        return simulator, handler

    return _build


@pytest.fixture()
def run_compare():
    """
    Run ``compare_logl`` with matplotlib mocked.

    :returns: A callable giving back the mock figure and the two mock axes.
    """

    def _run(simulator, handler, interactive: bool = False, **kwargs):
        fig, ax2d, ax1d = MagicMock(), MagicMock(), MagicMock()
        with (
            patch("mach3sbitools.diagnostics.compare_log.plt") as plt,
            patch(
                "mach3sbitools.diagnostics.compare_log.np.polyfit",
                return_value=(1.0, 0.0),
            ),
        ):
            plt.subplots.return_value = (fig, (ax2d, ax1d))
            plt.isinteractive.return_value = interactive
            ax2d.hist2d.return_value = (None, None, None, MagicMock())
            compare_logl(simulator, handler, **kwargs)
        return fig, ax2d, ax1d

    return _run


class TestCompareLogl:
    def test_draws_both_likelihoods(self, mocks, run_compare):
        simulator, handler = mocks()
        _, ax2d, ax1d = run_compare(simulator, handler, n_samples=20)

        handler.sample_posterior.assert_called_once()
        handler.get_log_likelihood.assert_called_once()
        ax2d.hist2d.assert_called_once()
        assert ax1d.hist.call_count == 2

    def test_simulator_likelihood_is_evaluated_per_sample(self, mocks, run_compare):
        n_samples = 15
        simulator, handler = mocks(n_samples)
        run_compare(simulator, handler, n_samples=n_samples)
        assert simulator.simulator_wrapper.get_log_likelihood.call_count == n_samples

    def test_likelihood_range_sets_the_2d_bin_edges(self, mocks, run_compare):
        """The requested clip must reach the histogram, not just be accepted."""
        simulator, handler = mocks()
        _, ax2d, _ = run_compare(
            simulator, handler, n_samples=20, likelihood_range=(-3.0, 3.0), n_bins=11
        )

        bins = ax2d.hist2d.call_args.kwargs["bins"][0]
        assert len(bins) == 11
        assert bins[0] == pytest.approx(-3.0)
        assert bins[-1] == pytest.approx(3.0)

    def test_save_path_writes_the_figure(self, mocks, run_compare, tmp_path):
        simulator, handler = mocks()
        fig, _, _ = run_compare(
            simulator, handler, n_samples=20, save_path=tmp_path / "compare.png"
        )
        fig.savefig.assert_called_once()

    def test_no_save_path_writes_nothing(self, mocks, run_compare):
        simulator, handler = mocks()
        fig, _, _ = run_compare(simulator, handler, n_samples=20)
        fig.savefig.assert_not_called()

    @pytest.mark.parametrize("interactive", [True, False])
    def test_figure_is_shown_only_when_interactive(
        self, mocks, run_compare, interactive
    ):
        simulator, handler = mocks()
        fig, _, _ = run_compare(
            simulator, handler, n_samples=20, interactive=interactive
        )
        assert fig.show.called is interactive
