"""
Block size sensitivity study: find optimal B for block bootstrap.

This script tests multiple block sizes and identifies which one
produces the best covariance estimate.
"""
import numpy as np
import matplotlib.pyplot as plt
from mock_sf_covariance_v2 import (
    generate_irregular_times, simulate_drw_lightcurves,
    estimate_true_covariance, bootstrap_sf_covariance,
    bootstrap_sf_covariance_block
)


def run_sensitivity_study(block_sizes):
    """
    Run mock experiment with multiple block sizes.

    Parameters
    ----------
    block_sizes : list of int
        Block sizes to test.

    Returns
    -------
    results : dict
        Results for all block sizes.
    """
    np.random.seed(42)
    rng = np.random.default_rng(42)

    # Setup (same as mock experiment)
    n_epochs = 80
    t_max = 5.0
    times = generate_irregular_times(n_epochs, t_max, rng)

    sigma_drw = 0.2
    tau_drw = 0.5
    sigma_phot = 0.01

    n_lags = 8
    tau_min = 0.02
    tau_max = 2.0
    lags = np.logspace(np.log10(tau_min), np.log10(tau_max), n_lags)
    lag_width = 0.15

    # True covariance from 1000 MC realizations
    n_realizations = 1000
    all_mags = simulate_drw_lightcurves(
        times, sigma_drw, tau_drw, n_realizations, rng, sigma_phot=sigma_phot
    )
    cov_true, _, _ = estimate_true_covariance(
        times, all_mags, lags, lag_width, z=0.0, sigma_phot=sigma_phot
    )

    # Single lightcurve for bootstrap
    mags_data = all_mags[0]
    n_bootstrap = 500

    # Epoch bootstrap (baseline)
    cov_boot_epoch, _, _ = bootstrap_sf_covariance(
        times, mags_data, lags, lag_width, n_bootstrap, rng,
        z=0.0, sigma_phot=sigma_phot
    )

    results = {
        "times": times,
        "lags": lags,
        "cov_true": cov_true,
        "cov_boot_epoch": cov_boot_epoch,
        "block_sizes": block_sizes,
        "epoch_metrics": {},
        "block_metrics": {},
    }

    # Baseline (epoch) metrics
    diag_true = np.diag(cov_true)
    diag_epoch = np.diag(cov_boot_epoch)
    results["epoch_metrics"]["lag0_ratio"] = diag_epoch[0] / diag_true[0]
    results["epoch_metrics"]["var_rmse"] = np.sqrt(np.mean((diag_epoch - diag_true)**2))

    # Test each block size
    print("\nTesting block sizes:", block_sizes)
    print("-" * 70)

    for B in block_sizes:
        print(f"B = {B:3d}...", end=" ", flush=True)
        cov_block, _, _ = bootstrap_sf_covariance_block(
            times, mags_data, lags, lag_width, n_bootstrap, rng,
            z=0.0, sigma_phot=sigma_phot, block_size=B
        )

        diag_block = np.diag(cov_block)
        lag0_ratio = diag_block[0] / diag_true[0]
        var_rmse = np.sqrt(np.mean((diag_block - diag_true)**2))

        results["block_metrics"][B] = {
            "cov": cov_block,
            "lag0_ratio": lag0_ratio,
            "var_rmse": var_rmse,
        }

        improvement_lag0 = results["epoch_metrics"]["lag0_ratio"] / lag0_ratio
        improvement_var = results["epoch_metrics"]["var_rmse"] / var_rmse

        print(f"lag0={lag0_ratio:.2f}× (→{improvement_lag0:.1f}× improvement), "
              f"var_rmse={var_rmse:.4f} (→{improvement_var:.1f}× improvement)")

    return results


def plot_sensitivity_results(results, save_dir="plots"):
    """Create diagnostic plots for block size sensitivity."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    block_sizes = results["block_sizes"]
    lags = results["lags"]
    cov_true = results["cov_true"]

    diag_true = np.diag(cov_true)
    diag_epoch = np.diag(results["cov_boot_epoch"])

    # Extract metrics
    lag0_ratios = [results["epoch_metrics"]["lag0_ratio"]]
    var_rmses = [results["epoch_metrics"]["var_rmse"]]

    for B in block_sizes:
        lag0_ratios.append(results["block_metrics"][B]["lag0_ratio"])
        var_rmses.append(results["block_metrics"][B]["var_rmse"])

    # Plot 1: Lag-0 variance ratio vs. block size
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = ["Epoch"] + [f"B={B}" for B in block_sizes]
    colors = ["red"] + ["green" if i < len(block_sizes)//2 else "blue" for i in range(len(block_sizes))]

    bars = ax.bar(range(len(lag0_ratios)), lag0_ratios, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='True value')
    ax.axhline(y=2.0, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Target (<2×)')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Lag-0 variance ratio', fontsize=12)
    ax.set_title('Block size sensitivity: Lag-0 variance overestimate', fontsize=13)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=11)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, lag0_ratios)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}×', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_sensitivity_lag0.png", dpi=100)
    plt.close()

    # Plot 2: Variance RMSE vs. block size
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(range(len(var_rmses)), var_rmses, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.axhline(y=0.020, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Target (<0.020)')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Variance RMSE', fontsize=12)
    ax.set_title('Block size sensitivity: Variance estimation error', fontsize=13)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=11)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, var_rmses)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_sensitivity_var_rmse.png", dpi=100)
    plt.close()

    # Plot 3: Covariance diagonal for selected block sizes
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(range(len(lags)), diag_true, 'o-', label='True', linewidth=2.5, markersize=8, color='black')
    ax.plot(range(len(lags)), diag_epoch, 's--', label='Epoch bootstrap', linewidth=2, markersize=7, color='red')

    # Plot best and worst block bootstraps
    if block_sizes:
        best_B = min(block_sizes, key=lambda b: results["block_metrics"][b]["var_rmse"])
        diag_best = np.diag(results["block_metrics"][best_B]["cov"])
        ax.plot(range(len(lags)), diag_best, '^--', label=f'Block (B={best_B}) [best]',
                linewidth=2, markersize=7, color='green')

        if len(block_sizes) > 1:
            worst_B = max(block_sizes, key=lambda b: results["block_metrics"][b]["var_rmse"])
            diag_worst = np.diag(results["block_metrics"][worst_B]["cov"])
            ax.plot(range(len(lags)), diag_worst, 'v--', label=f'Block (B={worst_B}) [worst]',
                    linewidth=2, markersize=7, color='purple')

    ax.set_xlabel('Lag index', fontsize=12)
    ax.set_ylabel('Variance of log SF²', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_title('Variance profile: Epoch vs. Block bootstrap (best/worst)', fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_sensitivity_variance_profile.png", dpi=100)
    plt.close()

    print(f"\nSensitivity plots saved to {save_dir}/")
    print(f"  - fig_sensitivity_lag0.png")
    print(f"  - fig_sensitivity_var_rmse.png")
    print(f"  - fig_sensitivity_variance_profile.png")


def print_summary(results):
    """Print summary of sensitivity study."""
    print("\n" + "=" * 70)
    print("BLOCK SIZE SENSITIVITY STUDY SUMMARY")
    print("=" * 70)

    block_sizes = results["block_sizes"]
    epoch_lag0 = results["epoch_metrics"]["lag0_ratio"]
    epoch_var = results["epoch_metrics"]["var_rmse"]

    print(f"\nBaseline (epoch bootstrap):")
    print(f"  Lag-0 variance ratio: {epoch_lag0:.2f}×")
    print(f"  Variance RMSE:        {epoch_var:.6f}")

    print(f"\nBlock bootstrap results:")
    print("-" * 70)
    print(f"{'Block Size':>11} | {'Lag-0 ratio':>12} | {'Improvement':>11} | "
          f"{'Var RMSE':>10} | {'Improvement':>11}")
    print("-" * 70)

    best_B = None
    best_score = float('inf')

    for B in block_sizes:
        metrics = results["block_metrics"][B]
        lag0 = metrics["lag0_ratio"]
        var = metrics["var_rmse"]
        imp_lag0 = epoch_lag0 / lag0
        imp_var = epoch_var / var

        # Scoring: prefer lower lag-0 ratio and lower var RMSE
        score = (lag0 - 1.0)**2 + var**2  # prefer close to 1.0 and 0
        if score < best_score:
            best_score = score
            best_B = B

        print(f"B = {B:6d} | {lag0:6.2f}×{'':<5} | {imp_lag0:6.1f}×{'':<4} | "
              f"{var:.6f} | {imp_var:6.1f}×")

    print("-" * 70)
    print(f"\n✓ Recommended block size: B = {best_B}")
    metrics_best = results["block_metrics"][best_B]
    print(f"  Lag-0 variance ratio: {metrics_best['lag0_ratio']:.2f}× (improvement: {epoch_lag0 / metrics_best['lag0_ratio']:.1f}×)")
    print(f"  Variance RMSE:        {metrics_best['var_rmse']:.6f} (improvement: {epoch_var / metrics_best['var_rmse']:.1f}×)")
    print("=" * 70)

    return best_B


if __name__ == "__main__":
    # Test these block sizes
    block_sizes = [5, 10, 15, 20, 25, 30, 35, 40]

    results = run_sensitivity_study(block_sizes)
    best_B = print_summary(results)
    plot_sensitivity_results(results)

    print(f"\nNext step: Run `test_block_bootstrap.py` with block_size={best_B}")
    print("           in mock_sf_covariance_v2.py:mock_experiment()")
