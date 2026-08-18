import json
from pathlib import Path

import matplotlib.pyplot as plt

LABELS: dict[str, str] = {
    "my": "Авторская реализация",
    "cv": "OpenCV",
    "pil": "Pillow",
}

MARKERS: dict[str, str] = {
    "my": "o",
    "cv": "s",
    "pil": "^",
}


def plot_results(data_file: Path) -> None:
    data = json.loads(data_file.read_text(encoding="utf-8"))

    sizes = [item["size"] for item in data]

    plt.figure(figsize=(10, 6))

    for key, label in LABELS.items():
        plt.errorbar(
            sizes,
            [item[f"{label}_среднее_мс"] for item in data],
            yerr=[item[f"{label}_отклонение_мс"] for item in data],
            label=label,
            marker=MARKERS[key],
            capsize=5,
            linewidth=2,
        )

    plt.xscale("log", base=2)
    plt.yscale("log")

    plt.xlabel("Размер изображения (пиксели)", fontsize=12)
    plt.ylabel("Время выполнения (мс, логарифмическая шкала)", fontsize=12)
    plt.title("Сравнение производительности: свёртка изображений", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.7)

    output_directory = Path(__file__).parent / "images"
    output_directory.mkdir(exist_ok=True)
    output_file = output_directory / "performance.png"

    plt.savefig(output_file, dpi=300)
    print(f"График сохранен в {output_file}")
