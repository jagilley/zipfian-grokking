"""
Ablation study over Zipf exponents for grokking.

Runs experiments with different Zipf exponents across multiple seeds and tracks:
- Epochs to reach 50% test accuracy (grokking onset)
- Epochs to reach 95% test accuracy (grokking completion)
- Final accuracies
- Mean/std aggregation across seeds
"""

import subprocess
import pickle
import os
import numpy as np
import argparse
from pathlib import Path


def find_threshold_epoch(history, threshold=0.5, metric="test_acc"):
    """Find first epoch where metric crosses threshold."""
    epochs = history["epoch"]
    values = history[metric]

    for epoch, val in zip(epochs, values):
        if val >= threshold:
            return epoch
    return None  # Never reached


def run_experiment(zipf_exp, seed, args):
    """Run a single experiment with given Zipf exponent and seed."""
    output_dir = os.path.join(args.output_dir, f"zipf_{zipf_exp}", f"seed_{seed}")

    cmd = [
        "python", "train_grokking_zipfian.py",
        "--modulus", str(args.modulus),
        "--train_frac", str(args.train_frac),
        "--zipf_exponent", str(zipf_exp),
        "--zipf_mode", args.zipf_mode,
        "--hidden_dims", *[str(d) for d in args.hidden_dims],
        "--n_epochs", str(args.n_epochs),
        "--learning_rate", str(args.learning_rate),
        "--weight_decay", str(args.weight_decay),
        "--log_interval", str(args.log_interval),
        "--print_interval", str(args.print_interval),
        "--fourier_interval", str(args.fourier_interval),
        "--save_dir", output_dir,
        "--seed", str(seed),
    ]

    if args.weighted_loss:
        cmd.append("--weighted_loss")

    print(f"\n{'='*60}")
    print(f"Running Zipf exponent = {zipf_exp}, seed = {seed}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"WARNING: Experiment with zipf_exp={zipf_exp}, seed={seed} failed!")
        return None

    # Load results
    history_path = os.path.join(output_dir, "history.pkl")
    if os.path.exists(history_path):
        with open(history_path, "rb") as f:
            history = pickle.load(f)
        return history
    return None


def main():
    parser = argparse.ArgumentParser(description="Ablation over Zipf exponents")

    # Ablation parameters
    parser.add_argument("--zipf_exponents", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0, 1.5, 2.0],
                        help="Zipf exponents to test")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 256, 999, 2024],
                        help="Random seeds for multi-seed runs")
    parser.add_argument("--output_dir", type=str, default="./ablation_zipf",
                        help="Output directory for all experiments")

    # Experiment parameters (passed through)
    parser.add_argument("--modulus", type=int, default=97)
    parser.add_argument("--train_frac", type=float, default=0.3)
    parser.add_argument("--zipf_mode", type=str, default="pair_rank")
    parser.add_argument("--weighted_loss", action="store_true",
                        help="Use Zipfian-weighted loss")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--n_epochs", type=int, default=60000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--print_interval", type=int, default=10000)
    parser.add_argument("--fourier_interval", type=int, default=2000)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Run experiments across all seeds
    results = {}  # {zipf_exp: {"seeds": {seed: {...}}, "aggregate": {...}}}
    for zipf_exp in args.zipf_exponents:
        seed_results = {}
        for seed in args.seeds:
            history = run_experiment(zipf_exp, seed, args)
            if history is not None:
                seed_results[seed] = {
                    "history": history,
                    "epoch_to_50": find_threshold_epoch(history, 0.5, "test_acc"),
                    "epoch_to_95": find_threshold_epoch(history, 0.95, "test_acc"),
                    "final_train_acc": history["train_acc"][-1],
                    "final_test_acc": history["test_acc"][-1],
                }

        if seed_results:
            # Aggregate across seeds
            final_test_accs = [r["final_test_acc"] for r in seed_results.values()]
            epochs_50 = [r["epoch_to_50"] for r in seed_results.values() if r["epoch_to_50"] is not None]
            epochs_95 = [r["epoch_to_95"] for r in seed_results.values() if r["epoch_to_95"] is not None]

            results[zipf_exp] = {
                "seeds": seed_results,
                "aggregate": {
                    "final_test_acc_mean": np.mean(final_test_accs),
                    "final_test_acc_std": np.std(final_test_accs),
                    "epoch_to_50_mean": np.mean(epochs_50) if epochs_50 else None,
                    "epoch_to_50_std": np.std(epochs_50) if epochs_50 else None,
                    "epoch_to_50_frac": len(epochs_50) / len(seed_results),
                    "epoch_to_95_mean": np.mean(epochs_95) if epochs_95 else None,
                    "epoch_to_95_std": np.std(epochs_95) if epochs_95 else None,
                    "epoch_to_95_frac": len(epochs_95) / len(seed_results),
                    "n_seeds": len(seed_results),
                },
            }

    # Print summary
    n_seeds = len(args.seeds)
    print(f"\n{'='*80}")
    print(f"ABLATION SUMMARY ({n_seeds} seeds)")
    print(f"{'='*80}")
    print(f"{'Zipf Exp':>10} | {'Epoch@50%':>18} | {'Epoch@95%':>18} | {'Final Test Acc':>18}")
    print("-" * 80)

    for zipf_exp in sorted(results.keys()):
        agg = results[zipf_exp]["aggregate"]
        if agg["epoch_to_50_mean"] is not None:
            e50_str = f"{agg['epoch_to_50_mean']:.0f} +/- {agg['epoch_to_50_std']:.0f}"
            if agg["epoch_to_50_frac"] < 1.0:
                e50_str += f" ({agg['epoch_to_50_frac']*100:.0f}%)"
        else:
            e50_str = "Never"
        if agg["epoch_to_95_mean"] is not None:
            e95_str = f"{agg['epoch_to_95_mean']:.0f} +/- {agg['epoch_to_95_std']:.0f}"
            if agg["epoch_to_95_frac"] < 1.0:
                e95_str += f" ({agg['epoch_to_95_frac']*100:.0f}%)"
        else:
            e95_str = "Never"
        acc_str = f"{agg['final_test_acc_mean']:.4f} +/- {agg['final_test_acc_std']:.4f}"
        print(f"{zipf_exp:>10.1f} | {e50_str:>18} | {e95_str:>18} | {acc_str:>18}")

    # Save summary
    summary = {
        "zipf_exponents": args.zipf_exponents,
        "seeds": args.seeds,
        "results": results,
        "args": vars(args),
    }

    with open(os.path.join(args.output_dir, "ablation_summary.pkl"), "wb") as f:
        pickle.dump(summary, f)

    # Generate comparison plots
    generate_comparison_plots(results, args.output_dir)

    print(f"\nResults saved to {args.output_dir}")


def generate_comparison_plots(results, output_dir):
    """Generate comparison plots across all Zipf exponents with multi-seed aggregation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    zipf_exps = sorted(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Test accuracy curves (mean +/- std shading)
    ax = axes[0, 0]
    for zipf_exp in zipf_exps:
        seed_data = results[zipf_exp]["seeds"]
        # Interpolate all seeds onto a common epoch grid
        all_epochs = sorted(set(
            e for r in seed_data.values() for e in r["history"]["epoch"]
        ))
        all_accs = []
        for r in seed_data.values():
            interp_acc = np.interp(all_epochs, r["history"]["epoch"], r["history"]["test_acc"])
            all_accs.append(interp_acc)
        all_accs = np.array(all_accs)
        mean_acc = all_accs.mean(axis=0)
        std_acc = all_accs.std(axis=0)
        line, = ax.plot(all_epochs, mean_acc, label=f"Zipf exp={zipf_exp}", alpha=0.8)
        ax.fill_between(all_epochs, mean_acc - std_acc, mean_acc + std_acc,
                        alpha=0.15, color=line.get_color())
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label="50% threshold")
    ax.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, label="95% threshold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Test Accuracy vs Epoch (mean +/- std)")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Plot 2: Train accuracy curves (mean +/- std shading)
    ax = axes[0, 1]
    for zipf_exp in zipf_exps:
        seed_data = results[zipf_exp]["seeds"]
        all_epochs = sorted(set(
            e for r in seed_data.values() for e in r["history"]["epoch"]
        ))
        all_accs = []
        for r in seed_data.values():
            interp_acc = np.interp(all_epochs, r["history"]["epoch"], r["history"]["train_acc"])
            all_accs.append(interp_acc)
        all_accs = np.array(all_accs)
        mean_acc = all_accs.mean(axis=0)
        std_acc = all_accs.std(axis=0)
        line, = ax.plot(all_epochs, mean_acc, label=f"Zipf exp={zipf_exp}", alpha=0.8)
        ax.fill_between(all_epochs, mean_acc - std_acc, mean_acc + std_acc,
                        alpha=0.15, color=line.get_color())
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Accuracy")
    ax.set_title("Train Accuracy vs Epoch (mean +/- std)")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Plot 3: Epochs to thresholds bar chart (with error bars)
    ax = axes[1, 0]
    x = np.arange(len(zipf_exps))
    width = 0.35

    means_50, stds_50, means_95, stds_95 = [], [], [], []
    for zipf_exp in zipf_exps:
        agg = results[zipf_exp]["aggregate"]
        means_50.append(agg["epoch_to_50_mean"] if agg["epoch_to_50_mean"] is not None else float('nan'))
        stds_50.append(agg["epoch_to_50_std"] if agg["epoch_to_50_std"] is not None else 0)
        means_95.append(agg["epoch_to_95_mean"] if agg["epoch_to_95_mean"] is not None else float('nan'))
        stds_95.append(agg["epoch_to_95_std"] if agg["epoch_to_95_std"] is not None else 0)

    ax.bar(x - width/2, means_50, width, yerr=stds_50, capsize=4,
           label='Epoch @ 50%', color='steelblue')
    ax.bar(x + width/2, means_95, width, yerr=stds_95, capsize=4,
           label='Epoch @ 95%', color='coral')

    ax.set_xlabel("Zipf Exponent")
    ax.set_ylabel("Epochs")
    ax.set_title("Grokking Onset (50%) and Completion (95%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{z:.1f}" for z in zipf_exps])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Final test accuracy distribution (with error bars)
    ax = axes[1, 1]

    means_acc, stds_acc = [], []
    for zipf_exp in zipf_exps:
        agg = results[zipf_exp]["aggregate"]
        means_acc.append(agg["final_test_acc_mean"])
        stds_acc.append(agg["final_test_acc_std"])

    bars = ax.bar(x, means_acc, yerr=stds_acc, capsize=4, color='purple', alpha=0.7)
    ax.set_xlabel("Zipf Exponent")
    ax.set_ylabel("Final Test Accuracy")
    ax.set_title("Final Test Accuracy (mean +/- std)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{z:.1f}" for z in zipf_exps])
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, mean, std in zip(bars, means_acc, stds_acc):
        ax.annotate(f'{mean:.2f}\n({std:.2f})',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_comparison.png"), dpi=150)
    plt.close()

    print(f"Saved comparison plots to {output_dir}/ablation_comparison.png")


if __name__ == "__main__":
    main()
