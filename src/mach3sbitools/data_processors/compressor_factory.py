from .compressor_base import CompressorBase
from .pca import PCACompressor

_COMPRESSOR_REGISTRY = {"pca": PCACompressor}


def compressor_factory(compressor_name: str, **compressor_kwargs) -> CompressorBase:
    """
    Construct an unfitted compressor by name.

    :param compressor_name: Registered name, case-insensitive, e.g. ``"pca"``.
    :param compressor_kwargs: Forwarded to the compressor's constructor.
    :returns: A new, unfitted compressor.
    :raises KeyError: If *compressor_name* is not registered.
    """
    compressor = _COMPRESSOR_REGISTRY.get(compressor_name.lower())
    if compressor is None:
        raise KeyError(
            f"Compressor {compressor_name} not found. Please select {list(_COMPRESSOR_REGISTRY.keys())}"
        )
    return compressor(**compressor_kwargs)


def restore_compressor(compressor_state_dict: dict) -> CompressorBase:
    """
    Rebuild a fitted compressor from a checkpointed state dict.

    :param compressor_state_dict: Output of :meth:`CompressorBase.state_dict`.
    :returns: The restored, fitted compressor.
    :raises KeyError: If the state dict names an unregistered compressor.
    """
    compressor_name = compressor_state_dict.get("type", "")
    compressor = _COMPRESSOR_REGISTRY.get(compressor_name)
    if compressor is None:
        raise KeyError(
            f"Compressor {compressor_name} not found. Please select {list(_COMPRESSOR_REGISTRY.keys())}"
        )

    return compressor.from_state_dict(compressor_state_dict)
