import numpy as np
from numpy.typing import NDArray


def apply_border(
    image: NDArray[np.float32],
    pad: int,
    border_type: str,
) -> NDArray[np.float32]:

    # constant
    # дополнение краев нулями
    if border_type == "constant":
        return np.pad(
            image,
            ((pad, pad), (pad, pad), (0, 0)),
            mode="constant",
        ).astype(np.float32)

    # reflect
    # зеркальное отражение изображения
    if border_type == "reflect":
        return np.pad(
            image,
            ((pad, pad), (pad, pad), (0, 0)),
            mode="reflect",
        ).astype(np.float32)

    # replicate
    # повторение крайних пикселей
    if border_type == "replicate":
        return np.pad(
            image,
            ((pad, pad), (pad, pad), (0, 0)),
            mode="edge",
        ).astype(np.float32)

    # wrap
    # циклическое продолжение изображения
    if border_type == "wrap":
        return np.pad(
            image,
            ((pad, pad), (pad, pad), (0, 0)),
            mode="wrap",
        ).astype(np.float32)

    raise ValueError(f"Unknown border type: {border_type}")
