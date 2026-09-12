"""
Tests for the PCA compressors and the compressed training pipeline.
"""

import numpy as np
import pytest
import torch

from mach3sbitools.data_processors import compressor_factory, restore_compressor
from mach3sbitools.inference import InferenceHandler, inference_handler
from mach3sbitools.utils import TrainingConfig

N_FEATURES = 10
N_ROWS = 100


@pytest.fixture
def sample_data() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(N_ROWS, N_FEATURES)


@pytest.fixture
def make_compressor():
    """Build an unfitted PCA compressor."""

    def _make(n_components: int = 4, **kwargs):
        return compressor_factory("pca", n_components=n_components, **kwargs)

    return _make


@pytest.fixture
def fitted(make_compressor, sample_data):
    """Build a PCA compressor already fitted on :func:`sample_data`."""

    def _fit(n_components: int = 4, data: torch.Tensor | None = None, **kwargs):
        return make_compressor(n_components, **kwargs).fit(
            sample_data if data is None else data
        )

    return _fit


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────


class TestCompressorFactory:
    def test_creates_an_unfitted_pca(self, make_compressor):
        compressor = make_compressor(3)
        assert compressor.n_components == 3
        assert not compressor.is_fitted

    def test_name_is_case_insensitive(self):
        assert compressor_factory("PCA", n_components=2).n_components == 2

    def test_unknown_name_lists_the_options(self):
        with pytest.raises(KeyError, match="not found"):
            compressor_factory("does_not_exist")


# ─────────────────────────────────────────────────────────────────────────────
# Fitting
# ─────────────────────────────────────────────────────────────────────────────


class TestPCAFit:
    def test_fit_populates_the_components(self, fitted):
        compressor = fitted(4)
        assert compressor.is_fitted
        assert compressor.mean.shape == (N_FEATURES,)
        assert compressor.components.shape == (4, N_FEATURES)
        assert compressor.explained_variance.shape == (4,)

    def test_more_components_than_features_is_rejected(
        self, make_compressor, sample_data
    ):
        with pytest.raises(ValueError, match="exceeds n_features"):
            make_compressor(N_FEATURES + 1).fit(sample_data)

    def test_explained_variance_ratio_sums_to_one(self, fitted):
        ratio = fitted(5).explained_variance_ratio()
        torch.testing.assert_close(ratio.sum(), torch.tensor(1.0), atol=1e-6, rtol=0)

    def test_subsampling_limits_rows_used(self, fitted):
        assert fitted(3, subsample=25)._n_samples_fit == 25


@pytest.mark.parametrize(
    "method,args",
    [
        ("transform", (torch.randn(5, N_FEATURES),)),
        ("inverse_transform", (torch.randn(5, 3),)),
        ("explained_variance_ratio", ()),
    ],
)
def test_using_an_unfitted_compressor_raises(make_compressor, method, args):
    with pytest.raises(RuntimeError, match="fitted"):
        getattr(make_compressor(3), method)(*args)


# ─────────────────────────────────────────────────────────────────────────────
# Transform round-trips
# ─────────────────────────────────────────────────────────────────────────────


class TestPCATransform:
    def test_transform_reduces_the_feature_axis(self, fitted, sample_data):
        assert fitted(4).transform(sample_data).shape == (N_ROWS, 4)

    def test_inverse_transform_restores_the_shape(self, fitted, sample_data):
        compressor = fitted(4)
        reconstructed = compressor.inverse_transform(compressor.transform(sample_data))
        assert reconstructed.shape == sample_data.shape

    def test_full_rank_round_trip_is_lossless(self):
        """Keeping every component means nothing is thrown away."""
        torch.manual_seed(42)
        data = torch.randn(500, 5)
        compressor = compressor_factory("pca", n_components=5).fit(data)
        torch.testing.assert_close(
            compressor.inverse_transform(compressor.transform(data)),
            data,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_unbatched_row_stays_unbatched(self, fitted, sample_data):
        """A single row must not acquire a batch dimension on the way through."""
        compressor = fitted(3)
        row = sample_data[0]
        transformed = compressor.transform(row)
        assert transformed.shape == (3,)
        assert compressor.inverse_transform(transformed).shape == row.shape


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────────────────────────────────────


class TestCompressorStateDict:
    def test_round_trip_preserves_the_fit(self, fitted):
        compressor = fitted(4)
        restored = restore_compressor(compressor.state_dict())

        assert restored.n_components == compressor.n_components
        assert restored.subsample == compressor.subsample
        assert restored.niter == compressor.niter
        torch.testing.assert_close(restored.mean, compressor.mean)
        torch.testing.assert_close(restored.components, compressor.components)

    def test_restored_compressor_transforms_identically(self, fitted, sample_data):
        compressor = fitted(4)
        restored = restore_compressor(compressor.state_dict())
        torch.testing.assert_close(
            restored.transform(sample_data), compressor.transform(sample_data)
        )

    def test_unknown_type_is_rejected(self):
        with pytest.raises(KeyError, match="not found"):
            restore_compressor({"type": "not_a_real_compressor"})


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: the compressed training pipeline
# ─────────────────────────────────────────────────────────────────────────────

N_THETA_COMPONENTS = 8
N_X_COMPONENTS = 5


@pytest.fixture(scope="module")
def compressed_handler(train_handler, tmp_path_factory):
    """
    A handler trained with both theta and x compressed.

    :returns: Tuple of ``(handler, checkpoint_path)``.
    """
    checkpoint = tmp_path_factory.mktemp("compressed") / "model.ckpt"
    handler = train_handler(
        TrainingConfig(
            save_path=checkpoint,
            batch_size=256,
            max_epochs=2,
            stop_after_epochs=50,
            autosave_every=500,
            show_progress=False,
            num_workers=0,
        ),
        theta=N_THETA_COMPONENTS,
        x=N_X_COMPONENTS,
    )
    return handler, checkpoint


@pytest.mark.slow
class TestCompressedTrainingPipeline:
    """
    Compression has to be applied during training, not just at inference.

    Fitting compressors, storing them in the checkpoint, then training on raw
    data leaves the network sized for the raw dimensions while
    ``sample_posterior`` conditions it on compressed x.
    """

    def test_network_is_built_in_compressed_space(self, compressed_handler):
        handler, _ = compressed_handler
        assert handler._density_estimator.input_shape[0] == N_THETA_COMPONENTS
        assert handler._density_estimator.condition_shape[0] == N_X_COMPONENTS

    def test_samples_are_returned_in_original_space(
        self, compressed_handler, test_consts
    ):
        handler, _ = compressed_handler
        samples = handler.sample_posterior(
            200, np.ones(test_consts.x_dim, dtype=np.float32)
        )
        assert samples.shape == (200, test_consts.theta_dim)
        assert torch.all(torch.isfinite(samples))

    def test_samples_respect_prior_bounds(self, compressed_handler, test_consts):
        handler, _ = compressed_handler
        samples = handler.sample_posterior(
            200, np.ones(test_consts.x_dim, dtype=np.float32)
        ).cpu()
        lower = handler.prior.prior_data.lower_bounds.cpu()
        upper = handler.prior.prior_data.upper_bounds.cpu()
        assert torch.all(samples >= lower) and torch.all(samples <= upper)

    def test_checkpoint_round_trip_restores_compressors(
        self, compressed_handler, prior_save
    ):
        _, checkpoint = compressed_handler
        reloaded = InferenceHandler(prior_save)
        reloaded.load_posterior(checkpoint)

        assert reloaded._theta_compressor.n_components == N_THETA_COMPONENTS
        assert reloaded._x_compressor.n_components == N_X_COMPONENTS
        assert reloaded._density_estimator.input_shape[0] == N_THETA_COMPONENTS


class TestUncompressedTrainingPipeline:
    def test_network_matches_raw_dimensions(
        self, prior_save, merged_data_dir, posterior_config, test_consts
    ):
        """Without compressors the network keeps the dataset's dimensions."""
        handler = InferenceHandler(prior_save)
        handler.set_dataset(merged_data_dir)
        handler.create_posterior(posterior_config)
        estimator = handler._build_density_estimator_from_inference()

        assert estimator.input_shape[0] == test_consts.theta_dim
        assert estimator.condition_shape[0] == test_consts.x_dim

    def test_probe_batch_returns_whole_dataset_when_small(
        self, prior_save, merged_data_dir
    ):
        handler = InferenceHandler(prior_save)
        handler.set_dataset(merged_data_dir)
        theta, x = handler._probe_batch()

        assert len(theta) == len(handler.dataset)
        assert len(x) == len(handler.dataset)

    def test_probe_batch_samples_beyond_the_head_when_capped(
        self, prior_save, merged_data_dir, monkeypatch
    ):
        """
        A contiguous head slice would draw entirely from the first shard,
        biasing both the compressor fit and the z-score statistics.
        """
        monkeypatch.setattr(inference_handler, "_PROBE_ROWS", 100)

        handler = InferenceHandler(prior_save)
        handler.set_dataset(merged_data_dir)
        theta, _ = handler._probe_batch()

        assert len(theta) == 100
        head, _ = handler.dataset[:100]
        assert not torch.equal(theta, head)
