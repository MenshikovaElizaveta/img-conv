import numpy as np
from numpy.typing import NDArray

KERNELS: dict[str, NDArray[np.float32]] = {

    # identity
    # изображение без изменений
    "identity": np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.float32,
    ),

    # box blur
    # обычное размытие
    "box_blur": np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float32,
    ) / 9,

    # gaussian blur
    # размытие по Гауссу
    "gaussian_blur": np.array(
        [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1],
        ],
        dtype=np.float32,
    ) / 16,

    # sharpen
    # увеличение резкости
    "sharpen": np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    ),

    # edge detection
    # выделение границ
    "edge_detection": np.array(
        [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1],
        ],
        dtype=np.float32,
    ),

    # emboss
    # эффект тиснения
    "emboss": np.array(
        [
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2],
        ],
        dtype=np.float32,
    ),

    # sobel x
    # вертикальные границы
    "sobel_x": np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ],
        dtype=np.float32,
    ),

    # sobel y
    # горизонтальные границы
    "sobel_y": np.array(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ],
        dtype=np.float32,
    ),
}
