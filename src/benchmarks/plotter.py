import json
from pathlib import Path

import matplotlib.pyplot as plt

from benchmarks.measure import SIZES

LABELS: dict[str, str] = {
    "my": "My",
    "cv": "OpenCV",
    "pil": "Pillow",
}

MARKERS: dict[str, str] = {
    "my": "o",
    "cv": "s",
    "pil": "^",
}


def plot_results(data_file: Path) -> None:
    data = json.loads(
        data_file.read_text(encoding="utf-8")
    )

    output_directory = Path(__file__).parent / "images"
    output_directory.mkdir(exist_ok=True)

    color_data = [
        item
        for item in data
        if item["mode"] == "color"
    ]

    sizes = [
        item["size"]
        for item in color_data
    ]

    plt.figure(figsize=(10, 6))

    for key, label in LABELS.items():
        plt.errorbar(
            sizes,
            [
                item[f"{key}_mean_ms"]
                for item in color_data
            ],
            yerr=[
                item[f"{key}_std_ms"]
                for item in color_data
            ],
            label=label,
            marker=MARKERS[key],
            capsize=5,
            linewidth=2,
        )

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xticks(SIZES)

    plt.xlabel("Image size (pixels)", fontsize=12)
    plt.ylabel("Execution time (ms, logarithmic scale)", fontsize=12)
    plt.title("Color image processing performance", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.7)

    output_file = (
        output_directory
        / "color_performance.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
    )
    plt.close()

    grayscale_data = [
        item
        for item in data
        if item["mode"] == "grayscale"
    ]

    sizes = [
        item["size"]
        for item in grayscale_data
    ]

    plt.figure(figsize=(10, 6))

    for key, label in LABELS.items():
        plt.errorbar(
            sizes,
            [
                item[f"{key}_mean_ms"]
                for item in grayscale_data
            ],
            yerr=[
                item[f"{key}_std_ms"]
                for item in grayscale_data
            ],
            label=label,
            marker=MARKERS[key],
            capsize=5,
            linewidth=2,
        )

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xticks(SIZES)

    plt.xlabel("Image size (pixels)", fontsize=12)
    plt.ylabel("Execution time (ms, logarithmic scale)", fontsize=12)
    plt.title("Grayscale image processing performance", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.7)

    output_file = (
        output_directory
        / "grayscale_performance.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(10, 6))

    for mode in ["color", "grayscale"]:
        mode_data = [
            item
            for item in data
            if item["mode"] == mode
        ]

        sizes = [
            item["size"]
            for item in mode_data
        ]

        plt.plot(
            sizes,
            [
                item["my_vs_cv_ratio"]
                for item in mode_data
            ],
            marker="o",
            linewidth=2,
            label=f"My / OpenCV ({mode})",
        )

        plt.plot(
            sizes,
            [
                item["my_vs_pil_ratio"]
                for item in mode_data
            ],
            marker="s",
            linewidth=2,
            label=f"My / Pillow ({mode})",
        )

    plt.xscale("log", base=2)
    plt.xticks(SIZES)

    plt.xlabel("Image size (pixels)", fontsize=12)
    plt.ylabel("Execution time ratio", fontsize=12)
    plt.title("My implementation compared with OpenCV and Pillow", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.7)

    output_file = (
        output_directory / "speedup.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
    )
    plt.close()
