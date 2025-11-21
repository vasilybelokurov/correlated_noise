"""
Comprehensive test of block bootstrap vs. epoch bootstrap.

This script:
1. Runs the mock experiment with both bootstrap methods
2. Generates detailed comparison plots
3. Prints diagnostic statistics
4. Evaluates against acceptance criteria
"""
import numpy as np
import matplotlib.pyplot as plt
from mock_sf_covariance_v2 import mock_experiment


def plot_comparison_diagnostics(results, save_dir="plots"):
    """
    Create comprehensive diagnostic plots comparing epoch vs. block bootstrap.

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
    cov_boot_epoch = results["cov_boot_epoch"]
    cov_boot_block = results["cov_boot_block"]

    # Normalize correlations
    diag_true = np.sqrt(np.diag(cov_true))
    diag_epoch = np.sqrt(np.diag(cov_boot_epoch))
    diag_block = np.sqrt(np.diag(cov_boot_block))

    corr_true = cov_true / np.outer(diag_true, diag_true)
    corr_epoch = cov_boot_epoch / np.outer(diag_epoch, diag_epoch)
    corr_block = cov_boot_block / np.outer(diag_block, diag_block)

    # Eigenvalues
    evals_true = np.linalg.eigvalsh(cov_true)
    evals_epoch = np.linalg.eigvalsh(cov_boot_epoch)
    evals_block = np.linalg.eigvalsh(cov_boot_block)

    # --- Figure 1: Covariance heatmaps (3-panel) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    vmax = max(cov_true.max(), cov_boot_epoch.max(), cov_boot_block.max())

    im1 = axes[0].imshow(cov_true, cmap="viridis", aspect="auto", vmax=vmax)
    axes[0].set_title("True covariance (1000 MC realizations)")
    axes[0].set_xlabel("Lag index")
    axes[0].set_ylabel("Lag index")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(cov_boot_epoch, cmap="viridis", aspect="auto", vmax=vmax)
    axes[1].set_title("Epoch bootstrap (naive)")
    axes[1].set_xlabel("Lag index")
    axes[1].set_ylabel("Lag index")
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(cov_boot_block, cmap="viridis", aspect="auto", vmax=vmax)
    axes[2].set_title(f"Block bootstrap (B={results['block_size']})")
    axes[2].set_xlabel("Lag index")
    axes[2].set_ylabel("Lag index")
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_block_01_covariance_heatmaps.png", dpi=100)
    plt.close()

    # --- Figure 2: Correlation matrices (3-panel) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im1 = axes[0].imshow(corr_true, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    axes[0].set_title("True correlations")
    axes[0].set_xlabel("Lag index")
    axes[0].set_ylabel("Lag index")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(corr_epoch, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    axes[1].set_title("Epoch bootstrap correlations")
    axes[1].set_xlabel("Lag index")
    axes[1].set_ylabel("Lag index")
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(corr_block, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    axes[2].set_title(f"Block bootstrap correlations (B={results['block_size']})")
    axes[2].set_xlabel("Lag index")
    axes[2].set_ylabel("Lag index")
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_block_02_correlation_matrices.png", dpi=100)
    plt.close()

    # --- Figure 3: Eigenvalue spectra ---
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(range(len(evals_true)), evals_true, 'o-', label="True (MC)", linewidth=2.5, markersize=8, color='black')
    ax.plot(range(len(evals_epoch)), evals_epoch, 's--', label="Epoch bootstrap", linewidth=2, markersize=7, color='red')
    ax.plot(range(len(evals_block)), evals_block, '^--', label=f"Block bootstrap (B={results['block_size']})", linewidth=2, markersize=7, color='green')
    ax.set_xlabel("Eigenvalue index", fontsize=12)
    ax.set_ylabel("Eigenvalue", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    ax.set_title("Eigenvalue spectra comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_block_03_eigenvalues.png", dpi=100)
    plt.close()

    # --- Figure 4: Diagonal (variances) ---
    fig, ax = plt.subplots(figsize=(10, 6))

    diag_true_val = np.diag(cov_true)
    diag_epoch_val = np.diag(cov_boot_epoch)
    diag_block_val = np.diag(cov_boot_block)

    ax.plot(range(len(lags)), diag_true_val, 'o-', label="True variance", linewidth=2.5, markersize=8, color='black')
    ax.plot(range(len(lags)), diag_epoch_val, 's--', label="Epoch bootstrap", linewidth=2, markersize=7, color='red')
    ax.plot(range(len(lags)), diag_block_val, '^--', label=f"Block bootstrap (B={results['block_size']})", linewidth=2, markersize=7, color='green')
    ax.set_xlabel("Lag index", fontsize=12)
    ax.set_ylabel("Variance of log SF²", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_title("Diagonal (variance) comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_block_04_diagonal.png", dpi=100)
    plt.close()

    # --- Figure 5: Lag-0 variance detail ---
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['True', 'Epoch\nbootstrap', f'Block\nbootstrap\n(B={results["block_size"]})']
    vars_0 = [diag_true_val[0], diag_epoch_val[0], diag_block_val[0]]
    colors = ['black', 'red', 'green']

    bars = ax.bar(methods, vars_0, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Variance at lag-0', fontsize=12)
    ax.set_title('Lag-0 variance: Critical diagnostic', fontsize=13)
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, vars_0)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Highlight the true value as a reference
    true_val = diag_true_val[0]
    for i, val in enumerate(vars_0[1:], 1):
        ratio = val / true_val
        ax.text(i, max(vars_0) * 0.5, f'{ratio:.1f}×', fontsize=14, fontweight='bold',
                color='darkred' if ratio > 1.5 else 'darkgreen',
                ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_block_05_lag0_detail.png", dpi=100)
    plt.close()

    # --- Figure 6: Off-diagonal correlation scatter ---
    fig, ax = plt.subplots(figsize=(10, 8))

    mask = np.triu(np.ones_like(corr_true, dtype=bool), k=1)
    corr_true_offdiag = corr_true[mask]
    corr_epoch_offdiag = corr_epoch[mask]
    corr_block_offdiag = corr_block[mask]

    ax.scatter(corr_true_offdiag, corr_epoch_offdiag, alpha=0.5, s=60, label='Epoch bootstrap', color='red')
    ax.scatter(corr_true_offdiag, corr_block_offdiag, alpha=0.5, s=60, label=f'Block bootstrap (B={results["block_size"]})', color='green')

    # Add diagonal line (perfect agreement)
    lim = max(abs(corr_true_offdiag).max(), abs(corr_epoch_offdiag).max(), abs(corr_block_offdiag).max()) + 0.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=2, alpha=0.7, label='Perfect agreement')

    ax.set_xlabel("True correlation", fontsize=12)
    ax.set_ylabel("Bootstrap correlation", fontsize=12)
    ax.set_title("Off-diagonal correlation comparison", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_block_06_offdiag_scatter.png", dpi=100)
    plt.close()

    print(f"\nPlots saved to {save_dir}/")
    print(f"  - fig_block_01_covariance_heatmaps.png")
    print(f"  - fig_block_02_correlation_matrices.png")
    print(f"  - fig_block_03_eigenvalues.png")
    print(f"  - fig_block_04_diagonal.png")
    print(f"  - fig_block_05_lag0_detail.png")
    print(f"  - fig_block_06_offdiag_scatter.png")


def print_diagnostics(results):
    """Print detailed diagnostic statistics."""
    print("\n" + "=" * 80)
    print("BOOTSTRAP COVARIANCE VALIDATION: DETAILED DIAGNOSTICS")
    print("=" * 80)

    cov_true = results["cov_true"]
    cov_boot_epoch = results["cov_boot_epoch"]
    cov_boot_block = results["cov_boot_block"]
    lags = results["lags"]

    # 1. Mean absolute relative error
    print("\n1. MEAN ABSOLUTE RELATIVE ERROR IN COVARIANCE")
    print("-" * 80)
    err_epoch = results["mean_rel_err_epoch"]
    err_block = results["mean_rel_err_block"]
    print(f"  Epoch bootstrap:  {err_epoch:.4f} ({100*err_epoch:.1f}%)")
    print(f"  Block bootstrap:  {err_block:.4f} ({100*err_block:.1f}%)")
    print(f"  Improvement:      {err_epoch / err_block:.2f}×")
    target = 0.30
    status_block = "✓ PASS" if err_block < target else "✗ FAIL"
    print(f"  Target (<{target}):  {status_block}")

    # 2. Lag-0 variance
    print("\n2. LAG-0 VARIANCE ANALYSIS")
    print("-" * 80)
    var_true_0 = np.diag(cov_true)[0]
    var_epoch_0 = np.diag(cov_boot_epoch)[0]
    var_block_0 = np.diag(cov_boot_block)[0]
    ratio_epoch_0 = var_epoch_0 / var_true_0
    ratio_block_0 = var_block_0 / var_true_0
    print(f"  True variance:    {var_true_0:.6f}")
    print(f"  Epoch bootstrap:  {var_epoch_0:.6f} ({ratio_epoch_0:.2f}× true)")
    print(f"  Block bootstrap:  {var_block_0:.6f} ({ratio_block_0:.2f}× true)")
    target_ratio = 2.0
    status_block = "✓ PASS" if ratio_block_0 < target_ratio else "✗ FAIL"
    print(f"  Target (<{target_ratio}×):  {status_block}")

    # 3. Correlation structure
    print("\n3. CORRELATION MATRIX STRUCTURE")
    print("-" * 80)
    diag_true = np.sqrt(np.diag(cov_true))
    diag_epoch = np.sqrt(np.diag(cov_boot_epoch))
    diag_block = np.sqrt(np.diag(cov_boot_block))
    corr_true = cov_true / np.outer(diag_true, diag_true)
    corr_epoch = cov_boot_epoch / np.outer(diag_epoch, diag_epoch)
    corr_block = cov_boot_block / np.outer(diag_block, diag_block)

    mask = np.triu(np.ones_like(corr_true, dtype=bool), k=1)
    rmse_epoch_corr = np.sqrt(np.mean((corr_epoch[mask] - corr_true[mask])**2))
    rmse_block_corr = np.sqrt(np.mean((corr_block[mask] - corr_true[mask])**2))
    print(f"  RMSE in off-diag correlations:")
    print(f"    Epoch bootstrap:  {rmse_epoch_corr:.4f}")
    print(f"    Block bootstrap:  {rmse_block_corr:.4f}")
    print(f"    Improvement:      {rmse_epoch_corr / rmse_block_corr:.2f}×")

    # 4. Eigenvalue spectrum
    print("\n4. EIGENVALUE SPECTRUM")
    print("-" * 80)
    evals_true = np.linalg.eigvalsh(cov_true)
    evals_epoch = np.linalg.eigvalsh(cov_boot_epoch)
    evals_block = np.linalg.eigvalsh(cov_boot_block)

    rmse_epoch_evals = np.sqrt(np.mean((evals_epoch - evals_true)**2))
    rmse_block_evals = np.sqrt(np.mean((evals_block - evals_true)**2))
    print(f"  RMSE in eigenvalues:")
    print(f"    Epoch bootstrap:  {rmse_epoch_evals:.6f}")
    print(f"    Block bootstrap:  {rmse_block_evals:.6f}")
    print(f"    Improvement:      {rmse_epoch_evals / rmse_block_evals:.2f}×")

    # 5. Variance profile
    print("\n5. VARIANCE PROFILE ACROSS LAGS")
    print("-" * 80)
    var_true = np.diag(cov_true)
    var_epoch = np.diag(cov_boot_epoch)
    var_block = np.diag(cov_boot_block)
    rmse_epoch_var = np.sqrt(np.mean((var_epoch - var_true)**2))
    rmse_block_var = np.sqrt(np.mean((var_block - var_true)**2))
    print(f"  RMSE in variances:")
    print(f"    Epoch bootstrap:  {rmse_epoch_var:.6f}")
    print(f"    Block bootstrap:  {rmse_block_var:.6f}")
    print(f"    Improvement:      {rmse_epoch_var / rmse_block_var:.2f}×")

    # 6. Summary of acceptance criteria
    print("\n" + "=" * 80)
    print("ACCEPTANCE CRITERIA SUMMARY")
    print("=" * 80)
    criteria = {
        "Mean rel error < 30%": err_block < 0.30,
        "Lag-0 variance < 2×": ratio_block_0 < 2.0,
        "Improvement factor > 2×": (err_epoch / err_block) > 2.0,
        "Correlation RMSE reduction > 2×": (rmse_epoch_corr / rmse_block_corr) > 2.0,
        "Eigenvalue RMSE reduction > 2×": (rmse_epoch_evals / rmse_block_evals) > 2.0,
    }

    for criterion, passed in criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {criterion}")

    n_passed = sum(criteria.values())
    n_total = len(criteria)
    print(f"\nScore: {n_passed}/{n_total} criteria passed")
    print("=" * 80)


def main():
    """Run the full validation pipeline."""
    print("\n" + "=" * 80)
    print("BLOCK BOOTSTRAP VALIDATION TEST")
    print("=" * 80)

    # Run mock experiment
    print("\nRunning mock experiment...")
    results = mock_experiment()

    # Print summary
    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"Lags (rest-frame):           {results['lags']}")
    print(f"Mean abs rel error (epoch):  {results['mean_rel_err_epoch']:.4f}")
    print(f"Mean abs rel error (block):  {results['mean_rel_err_block']:.4f}")
    print(f"Improvement factor:          {results['mean_rel_err_epoch'] / results['mean_rel_err_block']:.2f}×")
    print(f"Block size:                  {results['block_size']} epochs")

    # Generate plots
    print("\nGenerating diagnostic plots...")
    plot_comparison_diagnostics(results)

    # Print detailed diagnostics
    print_diagnostics(results)


if __name__ == "__main__":
    main()
