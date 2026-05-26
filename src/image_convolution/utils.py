import numpy as np
from numpy.typing import NDArray
from PIL import Image


def load_image(path: str) -> NDArray[np.float32]:
    image = Image.open(path).convert("RGB")

    return np.array(image, dtype=np.float32)


def save_image(
    image: NDArray[np.float32],
    path: str,
) -> None:

    clipped = np.clip(image, 0, 255).astype(np.uint8)

    result = Image.fromarray(clipped)

    result.save(path)
