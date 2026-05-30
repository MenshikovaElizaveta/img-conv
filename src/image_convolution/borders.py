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
    h, w, c = image.shape
    out_h, out_w = h + 2 * pad, w + 2 * pad

    y_src = np.arange(out_h) - pad
    x_src = np.arange(out_w) - pad
    if border_type == "constant":
        result = np.zeros((out_h, out_w, c), dtype=np.float32)
        result[pad:pad + h, pad:pad + w] = image
        return result

    if border_type == "reflect":
        y_src = reflect_indices(y_src, h)
        x_src = reflect_indices(x_src, w)

    elif border_type == "replicate":
        y_src = np.clip(y_src, 0, h - 1)
        x_src = np.clip(x_src, 0, w - 1)

    elif border_type == "wrap":
        y_src %= h
        x_src %= w

    else:
        raise ValueError(f"Такого типа обработки края нет: {border_type}")

    y = y_src[:, np.newaxis]
    x = x_src[np.newaxis, :]

    return image[y, x]
