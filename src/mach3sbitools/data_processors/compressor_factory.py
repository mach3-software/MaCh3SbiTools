from .compressor_base import CompressorBase
from .pca import PCACompressor

_COMPRESSOR_REGISTRY = {"pca": PCACompressor}


def compressor_factory(compressor_name: str, **compressor_kwargs) -> CompressorBase:
    compressor = _COMPRESSOR_REGISTRY.get(compressor_name.lower())
    if compressor is None:
        raise KeyError(
            f"Compressor {compressor_name} not found. Please select {list(_COMPRESSOR_REGISTRY.keys())}"
        )
    return compressor(**compressor_kwargs)


def restore_compressor(compressor_state_dict: dict):
    compressor_name = compressor_state_dict.get("type", "")
    compressor = _COMPRESSOR_REGISTRY.get(compressor_name)
    if compressor is None:
        raise KeyError(
            f"Compressor {compressor_name} not found. Please select {list(_COMPRESSOR_REGISTRY.keys())}"
        )

    return compressor.from_state_dict(compressor_state_dict)
