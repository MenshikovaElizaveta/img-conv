import numpy as np
from numpy.typing import NDArray


def make_kernel(values: list[list[float]]) -> NDArray[np.float32]:
    return np.array(values, dtype=np.float32)


KERNELS: dict[str, NDArray[np.float32]] = {
    "identity": make_kernel([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]),

    "box_blur": make_kernel([
        [1 / 9, 1 / 9, 1 / 9],
        [1 / 9, 1 / 9, 1 / 9],
        [1 / 9, 1 / 9, 1 / 9],
    ]),

    "gaussian_blur": make_kernel([
        [1 / 16, 2 / 16, 1 / 16],
        [2 / 16, 4 / 16, 2 / 16],
        [1 / 16, 2 / 16, 1 / 16],
    ]),

    "sharpen": make_kernel([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ]),

    "edge_detection": make_kernel([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ]),

    "emboss": make_kernel([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2],
    ]),

    "sobel_x": make_kernel([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ]),

    "sobel_y": make_kernel([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ]),
}


def get_kernel(
    kernel_name: str,
) -> NDArray[np.float32]:
    if kernel_name not in KERNELS:
        raise Exception(
            f"Такого ядра не существует: {kernel_name}"
        )

    return KERNELS[kernel_name]
