import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
import os

# this script compares the performance of DALL·E 3 and GPT-4o in generating images
# it calculates the mean and confidence intervals for alignment and aesthetics scores
# and performs statistical tests to determine if one model is significantly better than the other

input_dir = os.path.join(os.getcwd(),"StatsResults")

# -----------------------------
# Config
# -----------------------------
CSV_IN = os.path.join(input_dir, "Group Evaluation - combined.csv")  # change if needed
CSV_OUT = os.path.join(os.getcwd(),"StatsResults","dalle3_vs_gpt4o_test_results.csv")

PLOT_ALIGN_MEAN = os.path.join(input_dir, "plot_alignment_95ci.png")
PLOT_AESTH_MEAN = os.path.join(input_dir, "plot_aesthetics_95ci.png")

PLOT_ALIGN_DIFF = os.path.join(input_dir, "paired_diff_alignment_95ci.png")
PLOT_AESTH_DIFF = os.path.join(input_dir, "paired_diff_aesthetics_95ci.png")

PLOT_DIFF_COMBINED = os.path.join(input_dir, "paired_diff_combined_95ci.png")  # optional convenience


# -----------------------------
# Helpers
# -----------------------------
def normalize_model_name(m: str) -> str:
    s = str(m).strip().lower()
    if "gpt" in s and "4o" in s:
        return "GPT-4o"
    if "dall" in s and "3" in s:
        return "DALL·E 3"
    return str(m)

def mean_ci_explicit(a, confidence=0.95):
    a = np.asarray(a, dtype=float)
    n = a.size
    m = float(np.mean(a))
    if n < 2:
        return m, m, m
    sd = float(np.std(a, ddof=1))
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf((1 + confidence) / 2.0, n - 1)
    h = se * tcrit
    return m, m - h, m + h

def plot_model_means_with_ci(gpt_vals, dalle_vals, metric_name, out_path):
    labels = ["GPT-4o", "DALL·E 3"]
    groups = [np.asarray(gpt_vals, dtype=float), np.asarray(dalle_vals, dtype=float)]
    means, lowers, uppers = [], [], []
    for arr in groups:
        m, lo, hi = mean_ci_explicit(arr, 0.95)
        means.append(m); lowers.append(lo); uppers.append(hi)
    yerr = np.vstack([np.array(means) - np.array(lowers),
                      np.array(uppers) - np.array(means)])
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(labels)), means, yerr=yerr, capsize=5, alpha=0.85)
    for i, m in enumerate(means):
        plt.text(i, m + 0.05, f"{m:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.xticks(range(len(labels)), labels)
    plt.ylabel("Score")
    plt.ylim(0, 5.2)
    plt.title(f"{metric_name.capitalize()} — Mean with 95% CI")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_paired_differences(gpt_vals, dalle_vals, metric_name, out_path):
    g = np.asarray(gpt_vals, dtype=float)
    d = np.asarray(dalle_vals, dtype=float)
    n = min(g.size, d.size)
    diffs = g[:n] - d[:n]
    m, lo, hi = mean_ci_explicit(diffs, 0.95)
    sorted_diffs = np.sort(diffs)
    plt.figure(figsize=(6, 4))
    plt.scatter(range(n), sorted_diffs, alpha=0.6, label="Differences")
    plt.axhline(m, linestyle='-', linewidth=2, label=f"Mean diff = {m:.2f}")
    plt.fill_between(range(n), lo, hi, alpha=0.2, label=f"95% CI [{lo:.2f}, {hi:.2f}]")
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.ylabel("Difference (GPT-4o − DALL·E 3)")
    plt.xlabel("Paired sample index (sorted)")
    plt.title(f"Paired Differences with 95% CI — {metric_name.capitalize()}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_paired_differences_combined(gpt_align, dalle_align, gpt_aesth, dalle_aesth, out_path):
    diffs_a = np.asarray(gpt_align, dtype=float) - np.asarray(dalle_align, dtype=float)
    diffs_b = np.asarray(gpt_aesth, dtype=float) - np.asarray(dalle_aesth, dtype=float)
    m_a, lo_a, hi_a = mean_ci_explicit(diffs_a, 0.95)
    m_b, lo_b, hi_b = mean_ci_explicit(diffs_b, 0.95)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, metric_name, diffs, m, lo, hi in [
        (axes[0], "Alignment", diffs_a, m_a, lo_a, hi_a),
        (axes[1], "Aesthetics", diffs_b, m_b, lo_b, hi_b),
    ]:
        sorted_diffs = np.sort(diffs)
        ax.scatter(range(sorted_diffs.size), sorted_diffs, alpha=0.6, label="Differences")
        ax.axhline(m, linestyle='-', linewidth=2, label=f"Mean diff = {m:.2f}")
        ax.fill_between(range(sorted_diffs.size), lo, hi, alpha=0.2, label=f"95% CI [{lo:.2f}, {hi:.2f}]")
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_title(metric_name, fontsize=12)
        ax.set_xlabel("Paired sample index (sorted)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Difference (GPT-4o − DALL·E 3)")
    fig.suptitle("Paired Differences with 95% CI", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    df = pd.read_csv(CSV_IN).rename(columns={"type": "model"})
    df.columns = [c.strip() for c in df.columns]
    tidy = df[["model", "AVG Alignment", "AVG Aesthetics"]].rename(
        columns={"AVG Alignment": "alignment", "AVG Aesthetics": "aesthetics"}
    )
    tidy["model"] = tidy["model"].apply(normalize_model_name)
    gpt = tidy[tidy["model"] == "GPT-4o"].reset_index(drop=True)
    dalle = tidy[tidy["model"] == "DALL·E 3"].reset_index(drop=True)
    rows = []
    for metric in ["alignment", "aesthetics"]:
        x = gpt[metric].to_numpy(float)
        y = dalle[metric].to_numpy(float)
        welch = stats.ttest_ind(x, y, equal_var=False, alternative="greater")
        rows.append({
            "metric": metric,
            "design": "Welch t-test (independent, one-sided)",
            "alt_hypothesis": "mean(GPT-4o) > mean(DALL·E 3)",
            "t_stat": float(welch.statistic),
            "p_value": float(welch.pvalue),
            "mean_gpt4o": float(np.mean(x)),
            "mean_dalle3": float(np.mean(y)),
            "mean_diff": float(np.mean(x) - np.mean(y)),
            "n_gpt4o": int(len(x)),
            "n_dalle3": int(len(y)),
        })
    if len(gpt) == len(dalle) and len(gpt) > 1:
        for metric in ["alignment", "aesthetics"]:
            x = gpt[metric].to_numpy(float)
            y = dalle[metric].to_numpy(float)
            diffs = x - y
            t_res = stats.ttest_rel(x, y, alternative="greater")
            rows.append({
                "metric": metric,
                "design": "Paired t-test (matched pairs, one-sided)",
                "alt_hypothesis": "mean(GPT-4o - DALL·E 3) > 0",
                "t_stat": float(t_res.statistic),
                "p_value": float(t_res.pvalue),
                "mean_gpt4o": float(np.mean(x)),
                "mean_dalle3": float(np.mean(y)),
                "mean_diff": float(np.mean(diffs)),
                "n_pairs": int(len(diffs)),
            })
            w_stat, w_p = stats.wilcoxon(diffs, alternative="greater", zero_method="wilcox", correction=False, mode="auto")
            rows.append({
                "metric": metric,
                "design": "Wilcoxon signed-rank (matched pairs, one-sided)",
                "alt_hypothesis": "median(GPT-4o - DALL·E 3) > 0",
                "statistic": float(w_stat),
                "p_value": float(w_p),
                "mean_diff": float(np.mean(diffs)),
                "n_pairs": int(len(diffs)),
            })
    results = pd.DataFrame(rows)
    results.to_csv(CSV_OUT, index=False)
    print(f"Saved results to: {CSV_OUT}")
    plot_model_means_with_ci(gpt["alignment"], dalle["alignment"], "alignment", PLOT_ALIGN_MEAN)
    plot_model_means_with_ci(gpt["aesthetics"], dalle["aesthetics"], "aesthetics", PLOT_AESTH_MEAN)
    plot_paired_differences(gpt["alignment"], dalle["alignment"], "alignment", PLOT_ALIGN_DIFF)
    plot_paired_differences(gpt["aesthetics"], dalle["aesthetics"], "aesthetics", PLOT_AESTH_DIFF)
    plot_paired_differences_combined(
        gpt["alignment"], dalle["alignment"], gpt["aesthetics"], dalle["aesthetics"], PLOT_DIFF_COMBINED
    )
