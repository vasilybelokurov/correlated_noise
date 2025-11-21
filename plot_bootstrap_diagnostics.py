"""
Diagnostic plots comparing true vs bootstrap SF covariance estimates.
"""
import numpy as np
import matplotlib.pyplot as plt
from mock_sf_covariance import mock_experiment


def plot_diagnostics(results, save_dir="plots"):
    """
    Create comprehensive diagnostic plots.

    Parameters
    ----------
    results : dict
        Output from mock_experiment()
    save_dir : str
        Directory to save plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    lags = results["lags"]
    cov_true = results["cov_true"]
    cov_boot = results["cov_boot"]
    rel_err = results["rel_err"]

    # Normalize correlations
    diag_true = np.sqrt(np.diag(cov_true))
    diag_boot = np.sqrt(np.diag(cov_boot))

    corr_true = cov_true / np.outer(diag_true, diag_true)
    corr_boot = cov_boot / np.outer(diag_boot, diag_boot)

    # Eigenvalues
    evals_true = np.linalg.eigvalsh(cov_true)
    evals_boot = np.linalg.eigvalsh(cov_boot)

    # --- Figure 1: Covariance heatmaps ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    vmax_cov = max(cov_true.max(), cov_boot.max())

    im1 = axes[0].imshow(cov_true, cmap="viridis", aspect="auto")
    axes[0].set_title("True covariance (from 1000 realizations)")
    axes[0].set_xlabel("Lag index")
    axes[0].set_ylabel("Lag index")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(cov_boot, cmap="viridis", aspect="auto")
    axes[1].set_title("Bootstrap covariance (500 resamples)")
    axes[1].set_xlabel("Lag index")
    axes[1].set_ylabel("Lag index")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_boot_covariance_heatmaps.png", dpi=100)
    plt.close()

    # --- Figure 2: Correlation matrices ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im1 = axes[0].imshow(corr_true, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    axes[0].set_title("True correlation matrix")
    axes[0].set_xlabel("Lag index")
    axes[0].set_ylabel("Lag index")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(corr_boot, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    axes[1].set_title("Bootstrap correlation matrix")
    axes[1].set_xlabel("Lag index")
    axes[1].set_ylabel("Lag index")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_boot_correlation_matrices.png", dpi=100)
    plt.close()

    # --- Figure 3: Eigenvalue spectra ---
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(range(len(evals_true)), evals_true, 'o-', label="True (MC)", linewidth=2, markersize=8)
    ax.plot(range(len(evals_boot)), evals_boot, 's--', label="Bootstrap", linewidth=2, markersize=8)
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_title("Eigenvalue spectra comparison")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_boot_eigenvalues.png", dpi=100)
    plt.close()

    # --- Figure 4: Diagonal comparison (variances) ---
    fig, ax = plt.subplots(figsize=(10, 6))

    diag_true = np.diag(cov_true)
    diag_boot = np.diag(cov_boot)

    ax.plot(range(len(lags)), diag_true, 'o-', label="True variance", linewidth=2, markersize=8)
    ax.plot(range(len(lags)), diag_boot, 's--', label="Bootstrap variance", linewidth=2, markersize=8)
    ax.set_xlabel("Lag index")
    ax.set_ylabel("Variance of log SF²")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_title("Diagonal (variance) comparison")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_boot_diagonal.png", dpi=100)
    plt.close()

    # --- Figure 5: Relative error heatmap ---
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(rel_err, cmap="RdBu_r", aspect="auto",
                   vmin=-1, vmax=1)
    ax.set_title(f"Relative error: (Boot - True) / True\nMean |rel error| = {results['mean_rel_err']:.3f}")
    ax.set_xlabel("Lag index")
    ax.set_ylabel("Lag index")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Relative error")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_boot_relative_error.png", dpi=100)
    plt.close()

    # --- Figure 6: Off-diagonal correlations ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract upper triangular elements
    mask = np.triu(np.ones_like(corr_true, dtype=bool), k=1)
    corr_true_offdiag = corr_true[mask]
    corr_boot_offdiag = corr_boot[mask]

    ax.scatter(corr_true_offdiag, corr_boot_offdiag, alpha=0.6, s=50)
    ax.set_xlabel("True correlation")
    ax.set_ylabel("Bootstrap correlation")
    ax.set_title("Off-diagonal correlation comparison")
    ax.grid(True, alpha=0.3)

    # Add diagonal line
    lim = max(abs(corr_true_offdiag).max(), abs(corr_boot_offdiag).max())
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=2, alpha=0.7, label="Perfect agreement")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_boot_offdiag_scatter.png", dpi=100)
    plt.close()

    print(f"Plots saved to {save_dir}/")
    print(f"  - fig_boot_covariance_heatmaps.png")
    print(f"  - fig_boot_correlation_matrices.png")
    print(f"  - fig_boot_eigenvalues.png")
    print(f"  - fig_boot_diagonal.png")
    print(f"  - fig_boot_relative_error.png")
    print(f"  - fig_boot_offdiag_scatter.png")


if __name__ == "__main__":
    results = mock_experiment()
    plot_diagnostics(results)

    print(f"\nMean absolute relative error: {results['mean_rel_err']:.4f}")
