from pathlib import Path

from .measure import run_experiments, save_results
from .plotter import plot_results


def main() -> None:
    benchmark_file = (
        Path(__file__).parent
        / "data"
        / "benchmark_raw.json"
    )
    
    results = run_experiments(benchmark_file)
    data_file = save_results(results)
    plot_results(data_file)


if __name__ == "__main__":
    main()
