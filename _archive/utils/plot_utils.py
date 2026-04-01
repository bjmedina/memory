import os
import numpy as np
import matplotlib.pyplot as plt


def equal_width_bar(to_plot, data=None, num_bins=5, label=""):
    """Create a histogram with equal-width bins."""
    bins = np.linspace(min(to_plot) if data is None else min(data), 
                       max(to_plot) if data is None else max(data), 
                       num_bins + 1)
    counts, edges = np.histogram(to_plot, bins=bins, density=True)
    plt.bar(edges[:-1], counts, width=np.diff(edges), edgecolor='black', alpha=0.7, label=label)


def plot_and_save_histogram(data, title, xlabel, filename, num_bins=50, save_dir="plots"):
    """Plot and save a histogram to disk."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    equal_width_bar(data, num_bins=num_bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


def plot_all_histograms(ratios, ua_mse, ta_mse, num_bins=50, save_path="plots/combined_hist.png", title="Histogram Overview"):
    """Plot and save grouped histograms for three metric types."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    for ax, data, title_, xlabel in zip(axs, [ratios, ua_mse, ta_mse],
                                        ["Score Ratio", "Full MSE", "Time-Averaged MSE"],
                                        ["Ratio", "Full MSE", "TA MSE"]):
        plt.sca(ax)
        equal_width_bar(data, num_bins=num_bins, label=title_)
        ax.set_title(title_)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()


def plot_torch_cochleagram(cochleagram):
    """Visualize a cochleagram stored as a PyTorch tensor."""
    arr = cochleagram.squeeze().detach().cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.imshow(arr, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Amplitude [dB]')
    plt.title("Cochleagram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_torch_corrs(corr_matrix):
    """Visualize a correlation matrix stored as a PyTorch tensor."""
    arr = corr_matrix.squeeze().detach().cpu().numpy()
    plt.figure(figsize=(8, 6))
    plt.matshow(arr, vmin=0, vmax=1)
    plt.colorbar(label='Pearson Correlation')
    plt.title("Segment Correlation")
    plt.tight_layout()
    plt.show()
