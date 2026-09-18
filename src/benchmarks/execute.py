from .measure import run_experiments, save_results
from .plotter import plot_results


def main() -> None:
    results = run_experiments()
    data_file = save_results(results)
    plot_results(data_file)


if __name__ == "__main__":
    main()
