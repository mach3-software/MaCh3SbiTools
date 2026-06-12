"""
Tests for compression.
"""

import pytest
import torch

from mach3sbitools.data_processors import (
    compressor_factory,
    restore_compressor,
)


@pytest.fixture
def sample_data():
    torch.manual_seed(42)
    return torch.randn(100, 10)


def test_compressor_factory_creates_pca():
    compressor = compressor_factory(
        "pca",
        n_components=3,
    )

    assert compressor.n_components == 3
    assert not compressor.is_fitted


def test_compressor_factory_invalid_name():
    with pytest.raises(KeyError) as exc:
        compressor_factory("does_not_exist")

    assert "not found" in str(exc.value)


def test_fit_sets_expected_attributes(sample_data):
    compressor = compressor_factory(
        "pca",
        n_components=4,
    )

    compressor.fit(sample_data)

    assert compressor.is_fitted
    assert compressor.mean is not None
    assert compressor.components is not None
    assert compressor.explained_variance is not None

    assert compressor.mean.shape == (10,)
    assert compressor.components.shape == (4, 10)
    assert compressor.explained_variance.shape == (4,)


def test_fit_raises_when_n_components_exceeds_features(sample_data):
    compressor = compressor_factory(
        "pca",
        n_components=11,
    )

    with pytest.raises(ValueError) as exc:
        compressor.fit(sample_data)

    assert "exceeds n_features" in str(exc.value)


def test_transform_before_fit_raises(sample_data):
    compressor = compressor_factory(
        "pca",
        n_components=3,
    )

    with pytest.raises(RuntimeError) as exc:
        compressor.transform(sample_data)

    assert "must be fitted before transform" in str(exc.value)


def test_inverse_transform_before_fit_raises():
    compressor = compressor_factory(
        "pca",
        n_components=3,
    )

    with pytest.raises(RuntimeError) as exc:
        compressor.inverse_transform(torch.randn(5, 3))

    assert "must be fitted before inverse_transform" in str(exc.value)


def test_explained_variance_ratio_before_fit_raises():
    compressor = compressor_factory(
        "pca",
        n_components=3,
    )

    with pytest.raises(RuntimeError) as exc:
        compressor.explained_variance_ratio()

    assert "not fitted" in str(exc.value)


def test_explained_variance_ratio_sums_to_one(sample_data):
    compressor = compressor_factory(
        "pca",
        n_components=5,
    )

    compressor.fit(sample_data)

    ratio = compressor.explained_variance_ratio()

    assert torch.isclose(
        ratio.sum(),
        torch.tensor(1.0),
        atol=1e-6,
    )


def test_transform_shape(sample_data):
    compressor = compressor_factory(
        "pca",
        n_components=4,
    )

    compressor.fit(sample_data)

    transformed = compressor.transform(sample_data)

    assert transformed.shape == (100, 4)


def test_inverse_transform_shape(sample_data):
    compressor = compressor_factory(
        "pca",
        n_components=4,
    )

    compressor.fit(sample_data)

    transformed = compressor.transform(sample_data)
    reconstructed = compressor.inverse_transform(transformed)

    assert reconstructed.shape == sample_data.shape


def test_transform_inverse_transform_round_trip_reasonable():
    torch.manual_seed(42)

    data = torch.randn(500, 5)

    compressor = compressor_factory(
        "pca",
        n_components=5,
    )

    compressor.fit(data)

    transformed = compressor.transform(data)
    reconstructed = compressor.inverse_transform(transformed)

    assert torch.allclose(
        data,
        reconstructed,
        atol=1e-4,
        rtol=1e-4,
    )


def test_1d_input_transform_and_inverse():
    torch.manual_seed(42)

    data = torch.randn(100, 6)

    compressor = compressor_factory(
        "pca",
        n_components=3,
    )

    compressor.fit(data)

    row = data[0]

    transformed = compressor.transform(row)
    reconstructed = compressor.inverse_transform(transformed)

    assert transformed.ndim == 1
    assert reconstructed.ndim == 1
    assert reconstructed.shape == row.shape


def test_state_dict_round_trip(sample_data):
    compressor = compressor_factory(
        "pca",
        n_components=4,
    )

    compressor.fit(sample_data)

    state = compressor.state_dict()

    restored = restore_compressor(state)

    assert restored.n_components == compressor.n_components
    assert restored.subsample == compressor.subsample
    assert restored.niter == compressor.niter

    assert torch.equal(restored.mean, compressor.mean)
    assert torch.equal(restored.components, compressor.components)
    assert torch.equal(
        restored.explained_variance,
        compressor.explained_variance,
    )


def test_restored_compressor_produces_same_transform(sample_data):
    compressor = compressor_factory(
        "pca",
        n_components=4,
    )

    compressor.fit(sample_data)

    restored = restore_compressor(
        compressor.state_dict(),
    )

    original_output = compressor.transform(sample_data)
    restored_output = restored.transform(sample_data)

    assert torch.allclose(
        original_output,
        restored_output,
    )


def test_restore_compressor_invalid_type():
    with pytest.raises(KeyError) as exc:
        restore_compressor(
            {
                "type": "not_a_real_compressor",
            }
        )

    assert "not found" in str(exc.value)


def test_subsampling_limits_rows_used():
    torch.manual_seed(42)

    data = torch.randn(100, 8)

    compressor = compressor_factory(
        "pca",
        n_components=3,
        subsample=25,
    )

    compressor.fit(data)

    assert compressor._n_samples_fit == 25
