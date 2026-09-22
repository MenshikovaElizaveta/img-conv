from typing import Any

import numpy as np
import pytest

from benchmarks.measure import (
    BORDER_TYPE,
    KERNEL_NAME,
    SIZES,
    run_my_impl,
    run_opencv,
    run_pillow,
)

IMPLEMENTATIONS = {
    "my": run_my_impl,
    "cv": run_opencv,
    "pil": run_pillow,
}


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("mode", ["color", "grayscale"])
@pytest.mark.parametrize(
    "implementation",
    IMPLEMENTATIONS,
)
def test_performance(
    benchmark: Any,
    size: int,
    mode: str,
    implementation: str,
) -> None:
    rng = np.random.default_rng(42)

    if mode == "color":
        image = rng.random(
            (size, size, 3),
            dtype=np.float32,
        )
    else:
        image = rng.random(
            (size, size),
            dtype=np.float32,
        )

    benchmark.extra_info["size"] = size
    benchmark.extra_info["mode"] = mode
    benchmark.extra_info["implementation"] = implementation

    result = benchmark(
        IMPLEMENTATIONS[implementation],
        image,
        KERNEL_NAME,
        BORDER_TYPE,
    )

    assert result.shape == image.shape
