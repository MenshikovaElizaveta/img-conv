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


def setup_plot(
    title: str,
    ylabel: str,
    logarithmic_y: bool = True,
) -> None:
    plt.xscale("log", base=2)
    plt.xticks(SIZES)

    if logarithmic_y:
        plt.yscale("log")

    plt.xlabel("Image size (pixels)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.7)


def plot_performance(
    data: list[dict[str, float | int | str | bool]],
    mode: str,
    output_file: Path,
    title: str,
) -> None:
    mode_data = [
        item
        for item in data
        if item["mode"] == mode
    ]

    sizes = [
        item["size"]
        for item in mode_data
    ]

    plt.figure(figsize=(10, 6))

    for key, label in LABELS.items():
        plt.errorbar(
            sizes,
            [
                item[f"{key}_mean_ms"]
                for item in mode_data
            ],
            yerr=[
                item[f"{key}_std_ms"]
                for item in mode_data
            ],
            label=label,
            marker=MARKERS[key],
            capsize=5,
            linewidth=2,
        )

    setup_plot(
        title,
        "Execution time (ms, logarithmic scale)",
    )

    plt.savefig(
        output_file,
        dpi=300,
    )
    plt.close()


def plot_results(data_file: Path) -> None:
    data = json.loads(
        data_file.read_text(encoding="utf-8")
    )

    output_directory = Path(__file__).parent / "images"
    output_directory.mkdir(exist_ok=True)

    plot_performance(
        data,
        "color",
        output_directory / "color_performance.png",
        "Color image processing performance",
    )

    plot_performance(
        data,
        "grayscale",
        output_directory / "grayscale_performance.png",
        "Grayscale image processing performance",
    )

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

    setup_plot(
        "My implementation compared with OpenCV and Pillow",
        "Execution time ratio",
        logarithmic_y=False,
    )

    output_file = (
        output_directory
        / "speedup.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
    )
    plt.close()
