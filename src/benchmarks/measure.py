import json
import pathlib

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter
from scipy import stats

from image_convolution.convolver import apply_convolution
from image_convolution.kernels import get_kernel

SIZES: list[int] = [1024, 2048, 4096]
KERNEL_NAME: str = "box_blur"
BORDER_TYPE: str = "constant"


def run_my_impl(
    image: NDArray[np.float32],
    kernel_name: str,
    border_type: str,
) -> NDArray[np.float32]:
    grayscale = image.ndim == 2

    if grayscale:
        image = image[..., np.newaxis]

    result = apply_convolution(
        image,
        kernel_name,
        border_type,
    )

    if grayscale:
        return result[..., 0]

    return result


def run_opencv(
    image: NDArray[np.float32],
    kernel_name: str,
    border_type: str,
) -> NDArray[np.float32]:
    kernel = get_kernel(kernel_name)

    borders: dict[str, int] = {
        "constant": cv2.BORDER_CONSTANT,
        "reflect": cv2.BORDER_REFLECT_101,
        "replicate": cv2.BORDER_REPLICATE,
        "wrap": cv2.BORDER_WRAP,
    }

    return np.asarray(
        cv2.filter2D(
            image,
            ddepth=cv2.CV_32F,
            kernel=kernel,
            borderType=borders[border_type],
        ),
        dtype=np.float32,
    )


def run_pillow(
    image: NDArray[np.float32],
    kernel_name: str,
    border_type: str,
) -> NDArray[np.float32]:
    pillow_image = Image.fromarray(
        (image * 255).astype(np.uint8)
    )

    kernel = get_kernel(kernel_name)

    filtered = pillow_image.filter(
        ImageFilter.Kernel(
            size=kernel.shape,
            kernel=kernel.flatten().tolist(),
            scale=1,
        )
    )

    return np.asarray(
        filtered,
        dtype=np.float32,
    ) / 255.0


def make_results(
    benchmark_file: pathlib.Path,
) -> list[dict[str, float | int | str | bool]]:
    data = json.loads(
        benchmark_file.read_text(
            encoding="utf-8"
        )
    )

    grouped: dict[
        tuple[int, str],
        dict[str, object],
    ] = {}

    for benchmark in data["benchmarks"]:
        info = benchmark["extra_info"]

        size = int(info["size"])
        mode = str(info["mode"])
        implementation = str(
            info["implementation"]
        )

        key = (size, mode)

        if key not in grouped:
            grouped[key] = {
                "size": size,
                "mode": mode,
                "samples": {},
            }

        samples = grouped[key]["samples"]

        if isinstance(samples, dict):
            samples[implementation] = (
                benchmark["stats"]["data"]
            )

    results: list[
        dict[str, float | int | str | bool]
    ] = []

    for item in grouped.values():
        samples = item["samples"]

        if not isinstance(samples, dict):
            continue

        my_samples = np.asarray(
            samples["my"],
            dtype=np.float64,
        )
        cv_samples = np.asarray(
            samples["cv"],
            dtype=np.float64,
        )
        pil_samples = np.asarray(
            samples["pil"],
            dtype=np.float64,
        )

        my_mean = float(
            np.mean(my_samples) * 1000
        )
        cv_mean = float(
            np.mean(cv_samples) * 1000
        )
        pil_mean = float(
            np.mean(pil_samples) * 1000
        )

        my_std = float(
            np.std(my_samples, ddof=1) * 1000
        )
        cv_std = float(
            np.std(cv_samples, ddof=1) * 1000
        )
        pil_std = float(
            np.std(pil_samples, ddof=1) * 1000
        )

        cv_p_value = float(
            stats.ttest_ind(
                my_samples,
                cv_samples,
                equal_var=False,
            ).pvalue
        )

        pil_p_value = float(
            stats.ttest_ind(
                my_samples,
                pil_samples,
                equal_var=False,
            ).pvalue
        )

        results.append(
            {
                "size": int(str(item["size"])),
                "mode": str(item["mode"]),
                "my_mean_ms": my_mean,
                "my_std_ms": my_std,
                "cv_mean_ms": cv_mean,
                "cv_std_ms": cv_std,
                "pil_mean_ms": pil_mean,
                "pil_std_ms": pil_std,
                "my_vs_cv_ratio": my_mean / cv_mean,
                "my_vs_pil_ratio": my_mean / pil_mean,
                "my_vs_cv_significant": cv_p_value < 0.05,
                "my_vs_pil_significant": pil_p_value < 0.05,
                "my_vs_cv_p_value": cv_p_value,
                "my_vs_pil_p_value": pil_p_value,
            }
        )

    return results


def run_experiments(
    benchmark_file: pathlib.Path,
) -> list[
    dict[str, float | int | str | bool]
]:
    return make_results(
        benchmark_file
    )


def save_results(
    results: list[
        dict[str, float | int | str | bool]
    ],
) -> pathlib.Path:
    output_file = (
        pathlib.Path(__file__).parent
        / "data"
        / "results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_file
