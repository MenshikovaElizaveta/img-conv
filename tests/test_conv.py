from pathlib import Path

import numpy as np

from src.image_convolution.convolver import apply_convolution
from src.image_convolution.utils import load_image


def test_convolution() -> None:
    cases = [
        (
            "frog.jpg",
            "identity",
            "constant",
            "frog_identity_constant.jpg",
        ),
        (
            "pastery.jpg",
            "box_blur",
            "reflect",
            "pastery_box_blur_reflect.jpg",
        ),
        (
            "sky.jpg",
            "gaussian_blur",
            "replicate",
            "sky_gaussian_blur_replicate.jpg",
        ),
        (
            "sky.jpg",
            "sharpen",
            "wrap",
            "sky_sharpen_wrap.jpg",
        ),
        (
            "sky.jpg",
            "edge_detection",
            "wrap",
            "sky_edge_detection_wrap.jpg",
        ),
        (
            "sky.jpg",
            "emboss",
            "wrap",
            "sky_emboss_wrap.jpg",
        ),
        (
            "sky.jpg",
            "sobel_x",
            "wrap",
            "sky_sobel_x_wrap.jpg",
        ),
        (
            "sky.jpg",
            "sobel_y",
            "wrap",
            "sky_sobel_y_wrap.jpg",
        ),
    ]

    for input_name, kernel, border, expected_name in cases:
        image = load_image(
            str(Path("tests/input") / input_name)
        )

        expected = load_image(
            str(Path("tests/output") / expected_name)
        )

        result = apply_convolution(
            image=image,
            kernel_name=kernel,
            border_type=border,
        )

        assert result.shape == expected.shape

        assert np.allclose(
            result,
            expected,
            atol=1.0,
        ), (
            f"Ошибка для kernel={kernel}, "
            f"border={border}"
        )
