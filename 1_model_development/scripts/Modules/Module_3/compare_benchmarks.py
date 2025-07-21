import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# --- Configuration ---
BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "benchmarks"
OUTPUT_DIR = BENCHMARK_DIR / "comparison_results"
EXPERIMENTS = {
    "2048_module2": "hdc_evaluation_2048_module2",
    "2048_balanced": "hdc_evaluation_2048_balanced",
    "4096_module2": "hdc_evaluation_4096_module2",
    "4096_balanced": "hdc_evaluation_4096_balanced",
}

def plot_histories(histories, title, filename):
    plt.figure(figsize=(12, 8))
    for name, history in histories.items():
        plt.plot(history['val_acc'], label=f'{name} Validation Accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

def main():
    print("--- Generating Benchmark Comparison Report ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_histories = {}
    all_benchmarks = []

    for name, directory in EXPERIMENTS.items():
        history_path = BENCHMARK_DIR / directory / "training_history.csv"
        benchmark_path = BENCHMARK_DIR / directory / "benchmark_results.json"

        if history_path.exists():
            all_histories[name] = pd.read_csv(history_path)
        else:
            print(f"Warning: History file not found for {name}")

        if benchmark_path.exists():
            with open(benchmark_path, 'r') as f:
                benchmark_data = json.load(f)
                benchmark_data['experiment'] = name
                all_benchmarks.append(benchmark_data)
        else:
            print(f"Warning: Benchmark file not found for {name}")

    # Plot comparisons
    histories_2048 = {k: v for k, v in all_histories.items() if "2048" in k}
    plot_histories(histories_2048, "2048 Dimension Model Comparison", OUTPUT_DIR / "comparison_2048.png")

    histories_4096 = {k: v for k, v in all_histories.items() if "4096" in k}
    plot_histories(histories_4096, "4096 Dimension Model Comparison", OUTPUT_DIR / "comparison_4096.png")

    # Create and save summary table
    if all_benchmarks:
        summary_df = pd.DataFrame(all_benchmarks)
        summary_df = summary_df[['experiment', 'dimension', 'accuracy_percent', 'model_size_kb', 'training_time_seconds']]
        summary_df = summary_df.sort_values(by="accuracy_percent", ascending=False)
        summary_df.to_csv(OUTPUT_DIR / "summary_benchmark_table.csv", index=False)
        print("\n--- Benchmark Summary Table ---")
        print(summary_df.to_string())
        print(f"\nSummary table saved to {OUTPUT_DIR / 'summary_benchmark_table.csv'}")
    else:
        print("No benchmark data found to create a summary table.")

    print("\n--- Comparison Report Generation Complete ---")

if __name__ == "__main__":
    main()
