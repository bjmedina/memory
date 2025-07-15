import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def ensure_dir(dir_path):
    """
    Creates a directory if it doesn't already exist.

    Args:
        dir_path (str): Path to the directory to create.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")

def plot_dprime_by_isi(df_per_subject, stimulus_set=None, show_mean_sem=True, save_path=None):
    plt.figure(figsize=(10, 6))

    for subject_id, subj_df in df_per_subject.groupby("subject"):
        plt.plot(subj_df["isi"], subj_df["d_prime"], marker='o', alpha=0.4)

    if show_mean_sem:
        grouped = df_per_subject.groupby("isi")["d_prime"]
        mean_d = grouped.mean()
        sem_d  = grouped.sem()

        plt.errorbar(mean_d.index, mean_d.values, yerr=sem_d.values,
                     fmt='-o', color='black', capsize=3, linewidth=2,
                     label="mean ± SEM")

    plt.xlabel("ISI (Interstimulus Interval)")
    plt.ylabel("d′ (sensitivity index)")
    title = f"{stimulus_set}: d′ vs ISI Across Participants" if stimulus_set else "d′ vs ISI Across Participants"
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_itemwise_split_half_scatter_df(
    df,
    label="hit",
    n_splits=100,
    seed=42,
    label_hits_fa=None,
    stimulus_set=None,
    save_path=None
):
    np.random.seed(seed)
    x_vals, y_vals, item_ids = [], [], []

    for _ in range(n_splits):
        indices = np.arange(df.shape[0])
        np.random.shuffle(indices)
        half = len(indices) // 2
        group1 = df.iloc[indices[:half], :]
        group2 = df.iloc[indices[half:], :]

        x = np.nanmean(group1.values, axis=0)
        y = np.nanmean(group2.values, axis=0)
        valid = ~(np.isnan(x) | np.isnan(y))

        for xi, yi, item in zip(x[valid], y[valid], df.columns[valid]):
            x_vals.append(xi)
            y_vals.append(yi)
            item_ids.append(item)

    r = pearsonr(x_vals, y_vals)[0] if len(x_vals) >= 2 else float("nan")

    plt.figure(figsize=(6, 6))
    color = "green" if label == "hit" else "red"
    plt.scatter(x_vals, y_vals, color=color, alpha=0.6)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel(f"{label} rate (Split 1)")
    plt.ylabel(f"{label} rate (Split 2)")
    title = f"{stimulus_set}: Item-wise Split-Half {label.capitalize()} Rate (r = {r:.2f})" if stimulus_set else f"Item-wise Split-Half: {label.capitalize()} Rate (r = {r:.2f})"
    plt.title(title)
    plt.axis("square")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_power_curve(sizes, means, stds, label="hit", stimulus_set=None, save_path=None):
    color = "green" if label == "hit" else "red"

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, means, label="Mean split-half reliability", color=color)
    plt.fill_between(sizes, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                     color=color, alpha=0.2, label="±1 std")

    plt.xlabel("Number of Participants")
    plt.ylabel("Split-Half Reliability")
    
    title = f"Split-Half Reliability vs. Sample Size ({label})"
    if stimulus_set:
        title = f"{stimulus_set}: " + title
    plt.title(title)

    plt.grid(True)
    plt.legend()
    plt.ylim([0,1])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()