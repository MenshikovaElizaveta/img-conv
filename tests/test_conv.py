from pathlib import Path

import numpy as np

from image_convolution.convolver import apply_convolution
from image_convolution.utils import load_image


def test_convolution() -> None:
    cases = [
        (
            "frog.png",
            "identity",
            "constant",
            "frog_identity_constant.png",
        ),
        (
            "pastery.png",
            "box_blur",
            "reflect",
            "pastery_box_blur_reflect.png",
        ),
        (
            "sky.png",
            "gaussian_blur",
            "replicate",
            "sky_gaussian_blur_replicate.png",
        ),
        (
            "sky.png",
            "sharpen",
            "wrap",
            "sky_sharpen_wrap.png",
        ),
        (
            "sky.png",
            "edge_detection",
            "wrap",
            "sky_edge_detection_wrap.png",
        ),
        (
            "sky.png",
            "emboss",
            "wrap",
            "sky_emboss_wrap.png",
        ),
        (
            "sky.png",
            "sobel_x",
            "wrap",
            "sky_sobel_x_wrap.png",
        ),
        (
            "sky.png",
            "sobel_y",
            "wrap",
            "sky_sobel_y_wrap.png",
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

        result = np.clip(result, 0.0, 255.0)

        assert np.allclose(
            result,
            expected,
            atol=1.0,
        ), (f"Ошибка для kernel={kernel}, border={border}")
