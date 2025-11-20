import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches

def make_covariance_exp(x, sigma, ell):
    """
    Exponential covariance:
    C_ij = sigma^2 * exp(-|x_i - x_j| / ell)
    """
    x = np.asarray(x)
    N = len(x)
    C = np.empty((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            C[i, j] = sigma**2 * np.exp(-abs(x[i] - x[j]) / ell)
    return C

def generate_mock_data(N=50, a_true=2.0, b_true=1.0,
                       x_min=0.0, x_max=1.0,
                       sigma=0.1, ell=0.2,
                       random_seed=123):
    """
    Generate mock data for y = a_true * x + b_true with correlated noise.
    """
    rng = np.random.default_rng(random_seed)
    x = np.linspace(x_min, x_max, N)
    y_true = a_true * x + b_true

    # Build covariance and draw correlated noise
    C = make_covariance_exp(x, sigma, ell)
    L = np.linalg.cholesky(C)
    z = rng.normal(size=N)
    noise = L @ z
    y_obs = y_true + noise
    return x, y_obs, y_true, C

def fit_line_correlated(x, y, C):
    """
    Fit y = a x + b using full covariance C.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    N = len(x)

    A = np.vstack([x, np.ones_like(x)]).T
    L = np.linalg.cholesky(C)

    A_tilde = np.linalg.solve(L, A)
    y_tilde = np.linalg.solve(L, y)

    ATA = A_tilde.T @ A_tilde
    ATy = A_tilde.T @ y_tilde

    cov_theta = np.linalg.inv(ATA)
    theta_hat = cov_theta @ ATy

    residuals = y - A @ theta_hat
    r_tilde = np.linalg.solve(L, residuals)
    chi2 = r_tilde @ r_tilde

    return theta_hat, cov_theta, chi2

def fit_line_naive(x, y, sigma_eff):
    """
    Naive fit: assume independent Gaussian errors.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    N = len(x)

    A = np.vstack([x, np.ones_like(x)]).T

    sigma_eff = np.asarray(sigma_eff)
    if sigma_eff.ndim == 0:
        sigma_vec = np.full(N, float(sigma_eff))
    else:
        assert len(sigma_eff) == N
        sigma_vec = sigma_eff

    A_w = A / sigma_vec[:, None]
    y_w = y / sigma_vec

    ATA = A_w.T @ A_w
    ATy = A_w.T @ y_w

    cov_theta = np.linalg.inv(ATA)
    theta_hat = cov_theta @ ATy

    residuals = y - A @ theta_hat
    chi2 = np.sum((residuals / sigma_vec)**2)

    return theta_hat, cov_theta, chi2

def confidence_ellipse(mean, cov, n_std=2.0, **kwargs):
    """
    Create an ellipse patch representing a confidence region.
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=2*n_std*ell_radius_x, height=2*n_std*ell_radius_y,
                      **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = plt.matplotlib.transforms.Affine2D() \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])
    ellipse.set_transform(transf + plt.gca().transData)
    return ellipse

def main():
    # True parameters and noise properties
    a_true = 2.0
    b_true = 1.0
    sigma_noise = 0.1
    ell = 0.2
    N = 50

    # Generate one mock data set
    x, y_obs, y_true, C = generate_mock_data(
        N=N,
        a_true=a_true,
        b_true=b_true,
        sigma=sigma_noise,
        ell=ell,
        random_seed=123,
    )

    # Correct fit with full covariance
    theta_corr, cov_corr, chi2_corr = fit_line_correlated(x, y_obs, C)
    a_corr, b_corr = theta_corr
    sigma_a_corr = np.sqrt(cov_corr[0, 0])
    sigma_b_corr = np.sqrt(cov_corr[1, 1])

    # Naive fit
    sigma_diag = np.sqrt(np.diag(C))
    sigma_eff = np.median(sigma_diag)

    theta_naive, cov_naive, chi2_naive = fit_line_naive(x, y_obs, sigma_eff)
    a_naive, b_naive = theta_naive
    sigma_a_naive = np.sqrt(cov_naive[0, 0])
    sigma_b_naive = np.sqrt(cov_naive[1, 1])

    dof = N - 2

    print("="*60)
    print("SINGLE REALIZATION COMPARISON")
    print("="*60)
    print("\nTrue parameters:")
    print(f"  a_true = {a_true:.4f}, b_true = {b_true:.4f}")
    print("\nCorrect (correlated) fit:")
    print(f"  a = {a_corr:.4f} +/- {sigma_a_corr:.4f}")
    print(f"  b = {b_corr:.4f} +/- {sigma_b_corr:.4f}")
    print(f"  χ²/dof = {chi2_corr:.2f}/{dof} = {chi2_corr/dof:.3f}")
    print("\nNaive (uncorrelated) fit:")
    print(f"  a = {a_naive:.4f} +/- {sigma_a_naive:.4f}")
    print(f"  b = {b_naive:.4f} +/- {sigma_b_naive:.4f}")
    print(f"  χ²/dof = {chi2_naive:.2f}/{dof} = {chi2_naive/dof:.3f}")
    print(f"\nUncertainty ratio (naive/correct):")
    print(f"  σ_a: {sigma_a_naive/sigma_a_corr:.2f}× smaller")
    print(f"  σ_b: {sigma_b_naive/sigma_b_corr:.2f}× smaller")
    print("="*60)

    # ========== FIGURE 1: Data and fits with confidence bands ==========
    fig, ax = plt.subplots(figsize=(10, 6))

    x_fine = np.linspace(x.min(), x.max(), 400)

    # Plot data
    ax.errorbar(x, y_obs, yerr=np.sqrt(np.diag(C)), fmt='o',
                color='gray', alpha=0.6, label='Data (error bars from diag(C))',
                markersize=5, capsize=3)

    # True line
    ax.plot(x_fine, a_true * x_fine + b_true, 'k--', linewidth=2, label='True line', zorder=5)

    # Correlated fit with uncertainty band
    y_fit_corr = a_corr * x_fine + b_corr
    pred_err_corr = np.sqrt(sigma_a_corr**2 * x_fine**2 +
                             sigma_b_corr**2 +
                             2 * cov_corr[0, 1] * x_fine)
    ax.plot(x_fine, y_fit_corr, 'C0-', linewidth=2, label='Fit (correlated)')
    ax.fill_between(x_fine, y_fit_corr - pred_err_corr, y_fit_corr + pred_err_corr,
                     alpha=0.2, color='C0', label='±1σ (correlated)')

    # Naive fit with uncertainty band
    y_fit_naive = a_naive * x_fine + b_naive
    pred_err_naive = np.sqrt(sigma_a_naive**2 * x_fine**2 +
                              sigma_b_naive**2 +
                              2 * cov_naive[0, 1] * x_fine)
    ax.plot(x_fine, y_fit_naive, 'C1-', linewidth=2, label='Fit (naive)')
    ax.fill_between(x_fine, y_fit_naive - pred_err_naive, y_fit_naive + pred_err_naive,
                     alpha=0.2, color='C1', label='±1σ (naive)')

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.set_title('Linear Fit Comparison: Correlated vs Naive Noise Model', fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("/Users/vasilybelokurov/IoA Dropbox/Dr V.A. Belokurov/Code/correlated_noise/plots/fig_fits.png", dpi=150)
    print("\nSaved: plots/fig_fits.png")
    plt.close(fig)

    # ========== FIGURE 2: Confidence ellipses in parameter space ==========
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot true parameters
    ax.plot(a_true, b_true, 'k*', markersize=20, label='True parameters', zorder=10)

    # Correlated fit with 1σ and 2σ ellipses
    ellipse_corr_1 = confidence_ellipse(theta_corr, cov_corr, n_std=1.0,
                                       edgecolor='C0', facecolor='C0', alpha=0.2,
                                       linewidth=2, label='Correct fit (1σ, 2σ)')
    ellipse_corr_2 = confidence_ellipse(theta_corr, cov_corr, n_std=2.0,
                                       edgecolor='C0', facecolor='none',
                                       linewidth=2, linestyle='--')
    ax.add_patch(ellipse_corr_1)
    ax.add_patch(ellipse_corr_2)
    ax.plot(a_corr, b_corr, 'C0s', markersize=10, label='Correct fit center')

    # Naive fit with 1σ and 2σ ellipses
    ellipse_naive_1 = confidence_ellipse(theta_naive, cov_naive, n_std=1.0,
                                        edgecolor='C1', facecolor='C1', alpha=0.2,
                                        linewidth=2, label='Naive fit (1σ, 2σ)')
    ellipse_naive_2 = confidence_ellipse(theta_naive, cov_naive, n_std=2.0,
                                        edgecolor='C1', facecolor='none',
                                        linewidth=2, linestyle='--')
    ax.add_patch(ellipse_naive_1)
    ax.add_patch(ellipse_naive_2)
    ax.plot(a_naive, b_naive, 'C1s', markersize=10, label='Naive fit center')

    ax.set_xlabel('Slope a', fontsize=12)
    ax.set_ylabel('Intercept b', fontsize=12)
    ax.set_title('Parameter Uncertainty: Correlated vs Naive Model', fontsize=13)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig("/Users/vasilybelokurov/IoA Dropbox/Dr V.A. Belokurov/Code/correlated_noise/plots/fig_ellipses.png", dpi=150)
    print("Saved: plots/fig_ellipses.png")
    plt.close(fig)

    # ========== FIGURE 3: Residuals and correlation structure ==========
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Residuals from correlated fit
    residuals_corr = y_obs - (a_corr * x + b_corr)
    axes[0, 0].scatter(x, residuals_corr, s=30, alpha=0.6, color='C0')
    axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('x', fontsize=11)
    axes[0, 0].set_ylabel('Residuals', fontsize=11)
    axes[0, 0].set_title('Correlated Fit Residuals', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)

    # Residuals from naive fit
    residuals_naive = y_obs - (a_naive * x + b_naive)
    axes[0, 1].scatter(x, residuals_naive, s=30, alpha=0.6, color='C1')
    axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('x', fontsize=11)
    axes[0, 1].set_ylabel('Residuals', fontsize=11)
    axes[0, 1].set_title('Naive Fit Residuals', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)

    # Covariance matrix heatmap
    im1 = axes[1, 0].imshow(C, cmap='RdBu_r', aspect='auto', vmin=-0.01, vmax=0.01)
    axes[1, 0].set_xlabel('i', fontsize=11)
    axes[1, 0].set_ylabel('j', fontsize=11)
    axes[1, 0].set_title('Noise Covariance Matrix C', fontsize=12)
    plt.colorbar(im1, ax=axes[1, 0])

    # Correlation matrix (C_ij / sqrt(C_ii * C_jj))
    corr_matrix = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    im2 = axes[1, 1].imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes[1, 1].set_xlabel('i', fontsize=11)
    axes[1, 1].set_ylabel('j', fontsize=11)
    axes[1, 1].set_title('Correlation Matrix ρ', fontsize=12)
    plt.colorbar(im2, ax=axes[1, 1])

    fig.tight_layout()
    fig.savefig("/Users/vasilybelokurov/IoA Dropbox/Dr V.A. Belokurov/Code/correlated_noise/plots/fig_residuals.png", dpi=150)
    print("Saved: plots/fig_residuals.png")
    plt.close(fig)

    # ========== FIGURE 4: Chi-squared comparison ==========
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['Correlated', 'Naive']
    chi2_values = [chi2_corr, chi2_naive]
    chi2_dof_values = [chi2_corr/dof, chi2_naive/dof]
    colors = ['C0', 'C1']

    x_pos = np.arange(len(methods))
    width = 0.35

    bars = ax.bar(x_pos, chi2_dof_values, width, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(1.0, color='k', linestyle='--', linewidth=2, label='Expected (χ²/dof = 1)')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, chi2_dof_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('χ²/dof', fontsize=12)
    ax.set_title('Goodness of Fit: χ²/dof Comparison', fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim([0, max(chi2_dof_values) * 1.3])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig("/Users/vasilybelokurov/IoA Dropbox/Dr V.A. Belokurov/Code/correlated_noise/plots/fig_chisq.png", dpi=150)
    print("Saved: plots/fig_chisq.png")
    plt.close(fig)

if __name__ == "__main__":
    main()
