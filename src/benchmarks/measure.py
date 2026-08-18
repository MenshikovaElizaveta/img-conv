import json
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter

from image_convolution.convolver import apply_convolution
from image_convolution.kernels import get_kernel

SIZES: list[int] = [128, 512, 1024, 2048]
REPEATS: int = 30
WARMUP: int = 5
KERNEL_NAME: str = "box_blur"
BORDER_TYPE: str = "constant"

LABELS: dict[str, str] = {
    "my": "Авторская реализация",
    "cv": "OpenCV",
    "pil": "Pillow",
}

AdapterFunc = Callable[
    [NDArray[np.float32], str, str], 
    NDArray[np.float32]
]


def run_my_impl(
    image: NDArray[np.float32],
    kernel_name: str,
    border_type: str,
) -> NDArray[np.float32]:
    return apply_convolution(image, kernel_name, border_type)


def run_opencv(
    image: NDArray[np.float32],
    kernel_name: str,
    border_type: str,
) -> NDArray[np.float32]:
    kernel_matrix = get_kernel(kernel_name)
    
    border_map: dict[str, int] = {
        "constant": cv2.BORDER_CONSTANT,
        "reflect": cv2.BORDER_REFLECT_101, 
        "replicate": cv2.BORDER_REPLICATE,
        "wrap": cv2.BORDER_WRAP,
    }
    opencv_border = border_map[border_type]
    
    result = cv2.filter2D(
        image,
        ddepth=cv2.CV_32F,
        kernel=kernel_matrix,
        borderType=opencv_border,
    )
    
    return np.asarray(result, dtype=np.float32)


def run_pillow(
    image: NDArray[np.float32],
    kernel_name: str,
    border_type: str,
) -> NDArray[np.float32]:
    pillow_image = Image.fromarray((image * 255).astype(np.uint8))
    
    kernel_matrix = get_kernel(kernel_name)
    kernel_size = kernel_matrix.shape[0]
    kernel_list = kernel_matrix.flatten().tolist()
    
    pillow_filter = ImageFilter.Kernel(
        size=(kernel_size, kernel_size),
        kernel=kernel_list,
        scale=1,
    )
    filtered_image = pillow_image.filter(pillow_filter)
    
    result = np.asarray(filtered_image, dtype=np.float32) / 255.0
    return result


def measure_time(
    func: AdapterFunc,
    image: NDArray[np.float32],
    kernel_name: str,
    border_type: str,
) -> dict[str, float]:
    for _ in range(WARMUP):
        func(image, kernel_name, border_type)
        
    execution_times: list[float] = []
    for _ in range(REPEATS):
        start_time = time.perf_counter()
        func(image, kernel_name, border_type)
        end_time = time.perf_counter()
        execution_times.append(end_time - start_time)
        
    times_array = np.array(execution_times)
    
    return {
        "mean_ms": float(np.mean(times_array) * 1000),
        "std_ms": float(np.std(times_array) * 1000),
    }


def run_experiments() -> list[dict[str, float | int | str]]:
    results: list[dict[str, float | int | str]] = []
    
    print("Начало замеров...")
    
    for size in SIZES:
        print(f"Размер {size}x{size}")
        synthetic_image = np.random.rand(size, size, 3).astype(np.float32)
        
        stats = {
            "my": measure_time(run_my_impl, synthetic_image, KERNEL_NAME, BORDER_TYPE),
            "cv": measure_time(run_opencv, synthetic_image, KERNEL_NAME, BORDER_TYPE),
            "pil": measure_time(run_pillow, synthetic_image, KERNEL_NAME, BORDER_TYPE),
        }
        
        row: dict[str, float | int | str] = {"size": size}
        for key, label in LABELS.items():
            row[f"{label}_среднее_мс"] = stats[key]["mean_ms"]
            row[f"{label}_отклонение_мс"] = stats[key]["std_ms"]
        results.append(row)
        
    return results


def save_results(results: list[dict[str, float | int | str]]) -> Path:
    output_directory = Path(__file__).parent / "data"
    output_directory.mkdir(exist_ok=True)
    output_file = output_directory / "results.json"
    
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=2, ensure_ascii=False)
        
    print(f"Результаты сохранены в {output_file}")
    return output_file
