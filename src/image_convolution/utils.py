import numpy as np
from numpy.typing import NDArray


def load_image(path: str) -> NDArray[np.float32]:
    return np.array([[0.0]], dtype=np.float32)

def save_image(image: NDArray[np.float32], path: str) -> None:
    pass
