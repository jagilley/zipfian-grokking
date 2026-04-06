import argparse
import json
import os
import pickle
from collections import defaultdict
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

TRANSFORMS = ["translation", "scaling", "quadratic", "random"]
PAIRINGS = list(combinations(TRANSFORMS, 2))


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def match_dir(results_dir, a, b):
    path = os.path.join(results_dir, f"{a}_vs_{b}")
    if os.path.isdir(path):
        return path
    path = os.path.join(results_dir, f"{b}_vs_{a}")
    if os.path.isdir(path):
        return path
    return None


def safe_final_prob(history, name):
    key = f"prob_{name}"
    if key not in history or len(history[key]) == 0:
        return None
    return float(history[key][-1])


def analyze_match(path, a, b):
    history = load_pickle(os.path.join(path, "history.pkl"))
    policy_history = load_pickle(os.path.join(path, "policy_history.pkl"))

    final_prob_a = safe_final_prob(history, a)
    final_prob_b = safe_final_prob(history, b)
    peak_test = float(max(history["test_acc"]))
    final_test = float(history["test_acc"][-1])
    final_inverse_loss = float(history["inverse_loss"][-1])
    peak_inverse_acc = float(max(history.get("inverse_acc", [0.0])))

    if final_prob_a is None or final_prob_b is None:
        winner = None
        margin = None
    else:
        margin = final_prob_a - final_prob_b
        winner = a if margin >= 0 else b

    cf_gaps = []
    for entry in policy_history.get("cf_losses", []):
        if not isinstance(entry, dict):
            continue
        ids = {0: "translation", 1: "scaling", 2: "quadratic", 3: "random"}
        reverse_ids = {v: k for k, v in ids.items()}
        if reverse_ids[a] in entry and reverse_ids[b] in entry:
            cf_gaps.append(float(entry[reverse_ids[b]] - entry[reverse_ids[a]]))

    mean_cf_gap = float(np.mean(cf_gaps)) if cf_gaps else None

    return {
        "path": path,
        "pair": [a, b],
        "winner": winner,
        "margin": margin,
        "final_probabilities": {a: final_prob_a, b: final_prob_b},
        "peak_test_acc": peak_test,
        "final_test_acc": final_test,
        "final_inverse_loss": final_inverse_loss,
        "peak_inverse_acc": peak_inverse_acc,
        "mean_counterfactual_gap": mean_cf_gap,
    }


def summarize(results_dir):
    matches = []
    wins = defaultdict(int)
    margins = defaultdict(list)
    peak_accs = defaultdict(list)
    final_accs = defaultdict(list)

    for a, b in PAIRINGS:
        path = match_dir(results_dir, a, b)
        if path is None:
            continue
        result = analyze_match(path, a, b)
        matches.append(result)

        if result["winner"] is not None:
            wins[result["winner"]] += 1
            margin = abs(result["margin"])
            wins[a] += 0
            wins[b] += 0
            margins[a].append(result["final_probabilities"].get(a, 0.0) or 0.0)
            margins[b].append(result["final_probabilities"].get(b, 0.0) or 0.0)
            peak_accs[a].append(result["peak_test_acc"])
            peak_accs[b].append(result["peak_test_acc"])
            final_accs[a].append(result["final_test_acc"])
            final_accs[b].append(result["final_test_acc"])

    ranking = []
    for name in TRANSFORMS:
        ranking.append({
            "transform": name,
            "wins": int(wins[name]),
            "mean_final_policy_mass": float(np.mean(margins[name])) if margins[name] else 0.0,
            "mean_peak_test_acc": float(np.mean(peak_accs[name])) if peak_accs[name] else 0.0,
            "mean_final_test_acc": float(np.mean(final_accs[name])) if final_accs[name] else 0.0,
        })

    ranking.sort(
        key=lambda x: (
            x["wins"],
            x["mean_final_policy_mass"],
            x["mean_peak_test_acc"],
            x["mean_final_test_acc"],
        ),
        reverse=True,
    )

    finalists = [r["transform"] for r in ranking[:2]]

    return {
        "results_dir": results_dir,
        "matches": matches,
        "ranking": ranking,
        "finalists": finalists,
    }


def write_scoreboard_text(summary, out_path):
    lines = []
    lines.append("Transform Tournament Scoreboard")
    lines.append("================================")
    lines.append("")
    lines.append("Ranking:")
    for i, row in enumerate(summary["ranking"], start=1):
        lines.append(
            f"{i}. {row['transform']}: wins={row['wins']}, "
            f"mean_policy_mass={row['mean_final_policy_mass']:.3f}, "
            f"mean_peak_test={row['mean_peak_test_acc']:.3f}, "
            f"mean_final_test={row['mean_final_test_acc']:.3f}"
        )
    lines.append("")
    lines.append(f"Finalists: {summary['finalists'][0]} vs {summary['finalists'][1]}")
    lines.append("")
    lines.append("Matches:")
    for m in summary["matches"]:
        lines.append(
            f"- {m['pair'][0]} vs {m['pair'][1]} -> winner={m['winner']}, "
            f"final_probs={m['final_probabilities']}, peak_test={m['peak_test_acc']:.3f}, "
            f"final_test={m['final_test_acc']:.3f}"
        )
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_summary(summary, out_path):
    transforms = [r["transform"] for r in summary["ranking"]]
    wins = [r["wins"] for r in summary["ranking"]]
    masses = [r["mean_final_policy_mass"] for r in summary["ranking"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].bar(transforms, wins, color="#4c78a8")
    axes[0].set_title("Round-robin wins")
    axes[0].set_ylabel("wins")
    axes[0].set_ylim(0, max(1, max(wins) + 0.5))

    axes[1].bar(transforms, masses, color="#f58518")
    axes[1].set_title("Mean final policy mass")
    axes[1].set_ylabel("probability")
    axes[1].set_ylim(0, 1.0)

    fig.suptitle("Transform tournament summary")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--write_json", action="store_true")
    args = parser.parse_args()

    summary = summarize(args.results_dir)

    scoreboard_path = os.path.join(args.results_dir, "scoreboard.txt")
    write_scoreboard_text(summary, scoreboard_path)
    plot_summary(summary, os.path.join(args.results_dir, "scoreboard.png"))

    print(json.dumps(summary, indent=2))

    if args.write_json:
        with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
