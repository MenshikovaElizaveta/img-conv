from typing import cast

import numpy as np
from numpy.typing import NDArray


def reflect_indices(
    indices: NDArray[np.int_],
    size: int,
) -> NDArray[np.int_]:
    period = 2 * size - 2

    indices = np.abs(indices)
    indices %= period

    return np.where(
        indices >= size,
        period - indices,
        indices,
    )


def apply_border(
    image: NDArray[np.float32],
    pad: int,
    border_type: str,
) -> NDArray[np.float32]:
    height, width, channels = image.shape
    padded_height, padded_width = height + 2 * pad, width + 2 * pad

    y_src = np.arange(padded_height) - pad
    x_src = np.arange(padded_width) - pad

    if border_type == "constant":
        result = np.zeros((padded_height, padded_width, channels), dtype=np.float32)
        result[pad:pad + height, pad:pad + width] = image
        return result

    if border_type == "reflect":
        y_src = reflect_indices(y_src, height)
        x_src = reflect_indices(x_src, width)

    elif border_type == "replicate":
        y_src = np.clip(y_src, 0, height - 1)
        x_src = np.clip(x_src, 0, width - 1)

    elif border_type == "wrap":
        y_src %= height
        x_src %= width

    else:
        raise ValueError(
            f"Такого типа обработки края нет: {border_type}"
        )

    y = y_src[:, np.newaxis]
    x = x_src[np.newaxis, :]

    return cast(
        NDArray[np.float32],
        np.asarray(
            image[y, x],
            dtype=np.float32,
        ),
    )
