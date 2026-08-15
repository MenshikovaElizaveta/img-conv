import numpy as np
from numpy.typing import NDArray

from .borders import apply_border
from .kernels import get_kernel


def apply_convolution(
    image: NDArray[np.float32],
    kernel_name: str,
    border_type: str,
) -> NDArray[np.float32]:
    if image.size == 0:
        raise ValueError("Изображение пустое")

    kernel = get_kernel(kernel_name)

    kernel_height, kernel_width = kernel.shape
    pad = kernel_height // 2

    padded = apply_border(
        image,
        pad,
        border_type,
    )

    height, width, channels = image.shape
    result = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            for channel in range(channels):
                window = padded[
                    y:y + kernel_height,
                    x:x + kernel_width,
                    channel,
                ]

                result[y, x, channel] = np.sum(
                    window * kernel
                )

    return result
