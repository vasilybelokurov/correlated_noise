import numpy as np
import matplotlib.pyplot as plt

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

def monte_carlo_study(n_realizations=100):
    """
    Run Monte Carlo study comparing correlated vs naive fits across many realizations.
    """
    # Setup
    a_true = 2.0
    b_true = 1.0
    sigma_noise = 0.1
    ell = 0.2
    N = 50

    # Pre-compute covariance (same for all realizations)
    x = np.linspace(0.0, 1.0, N)
    C = make_covariance_exp(x, sigma_noise, ell)
    sigma_diag = np.sqrt(np.diag(C))
    sigma_eff = np.median(sigma_diag)

    # Storage for results
    a_corr_list = []
    b_corr_list = []
    sigma_a_corr_list = []
    sigma_b_corr_list = []

    a_naive_list = []
    b_naive_list = []
    sigma_a_naive_list = []
    sigma_b_naive_list = []

    chi2_corr_list = []
    chi2_naive_list = []

    # Run realizations
    print(f"\nRunning {n_realizations} Monte Carlo realizations...")
    for i in range(n_realizations):
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_realizations} done")

        _, y_obs, _, _ = generate_mock_data(
            N=N,
            a_true=a_true,
            b_true=b_true,
            sigma=sigma_noise,
            ell=ell,
            random_seed=1000 + i,  # Different seed each time
        )

        # Correlated fit
        theta_corr, cov_corr, chi2_corr = fit_line_correlated(x, y_obs, C)
        a_corr_list.append(theta_corr[0])
        b_corr_list.append(theta_corr[1])
        sigma_a_corr_list.append(np.sqrt(cov_corr[0, 0]))
        sigma_b_corr_list.append(np.sqrt(cov_corr[1, 1]))
        chi2_corr_list.append(chi2_corr)

        # Naive fit
        theta_naive, cov_naive, chi2_naive = fit_line_naive(x, y_obs, sigma_eff)
        a_naive_list.append(theta_naive[0])
        b_naive_list.append(theta_naive[1])
        sigma_a_naive_list.append(np.sqrt(cov_naive[0, 0]))
        sigma_b_naive_list.append(np.sqrt(cov_naive[1, 1]))
        chi2_naive_list.append(chi2_naive)

    # Convert to arrays
    a_corr = np.array(a_corr_list)
    b_corr = np.array(b_corr_list)
    sigma_a_corr = np.array(sigma_a_corr_list)
    sigma_b_corr = np.array(sigma_b_corr_list)

    a_naive = np.array(a_naive_list)
    b_naive = np.array(b_naive_list)
    sigma_a_naive = np.array(sigma_a_naive_list)
    sigma_b_naive = np.array(sigma_b_naive_list)

    chi2_corr = np.array(chi2_corr_list)
    chi2_naive = np.array(chi2_naive_list)

    dof = N - 2

    # ========== PRINT SUMMARY STATISTICS ==========
    print("\n" + "="*70)
    print("MONTE CARLO SUMMARY (100 realizations)")
    print("="*70)

    print("\n--- CORRELATED FIT ---")
    print(f"Slope a:")
    print(f"  Mean:   {a_corr.mean():.4f} (true: {a_true:.4f})")
    print(f"  Std:    {a_corr.std():.4f}")
    print(f"  Bias:   {a_corr.mean() - a_true:.4f}")
    print(f"  Mean predicted σ: {sigma_a_corr.mean():.4f}")
    print(f"  Ratio (predicted/observed): {sigma_a_corr.mean() / a_corr.std():.3f}")

    print(f"\nIntercept b:")
    print(f"  Mean:   {b_corr.mean():.4f} (true: {b_true:.4f})")
    print(f"  Std:    {b_corr.std():.4f}")
    print(f"  Bias:   {b_corr.mean() - b_true:.4f}")
    print(f"  Mean predicted σ: {sigma_b_corr.mean():.4f}")
    print(f"  Ratio (predicted/observed): {sigma_b_corr.mean() / b_corr.std():.3f}")

    print(f"\nχ²/dof:")
    print(f"  Mean: {(chi2_corr/dof).mean():.3f} (expected: 1.000)")
    print(f"  Std:  {(chi2_corr/dof).std():.3f}")

    print("\n--- NAIVE FIT ---")
    print(f"Slope a:")
    print(f"  Mean:   {a_naive.mean():.4f} (true: {a_true:.4f})")
    print(f"  Std:    {a_naive.std():.4f}")
    print(f"  Bias:   {a_naive.mean() - a_true:.4f}")
    print(f"  Mean predicted σ: {sigma_a_naive.mean():.4f}")
    print(f"  Ratio (predicted/observed): {sigma_a_naive.mean() / a_naive.std():.3f}")

    print(f"\nIntercept b:")
    print(f"  Mean:   {b_naive.mean():.4f} (true: {b_true:.4f})")
    print(f"  Std:    {b_naive.std():.4f}")
    print(f"  Bias:   {b_naive.mean() - b_true:.4f}")
    print(f"  Mean predicted σ: {sigma_b_naive.mean():.4f}")
    print(f"  Ratio (predicted/observed): {sigma_b_naive.mean() / b_naive.std():.3f}")

    print(f"\nχ²/dof:")
    print(f"  Mean: {(chi2_naive/dof).mean():.3f} (expected: 1.000)")
    print(f"  Std:  {(chi2_naive/dof).std():.3f}")

    print("\n--- COMPARISON ---")
    print(f"Empirical uncertainty in a:")
    print(f"  Correlated: {a_corr.std():.4f}")
    print(f"  Naive:      {a_naive.std():.4f}")
    print(f"  Ratio (naive/corr): {a_naive.std() / a_corr.std():.3f}×")

    print(f"\nEmpirical uncertainty in b:")
    print(f"  Correlated: {b_corr.std():.4f}")
    print(f"  Naive:      {b_naive.std():.4f}")
    print(f"  Ratio (naive/corr): {b_naive.std() / b_corr.std():.3f}×")
    print("="*70)

    return {
        'a_corr': a_corr, 'b_corr': b_corr, 'sigma_a_corr': sigma_a_corr, 'sigma_b_corr': sigma_b_corr,
        'a_naive': a_naive, 'b_naive': b_naive, 'sigma_a_naive': sigma_a_naive, 'sigma_b_naive': sigma_b_naive,
        'chi2_corr': chi2_corr, 'chi2_naive': chi2_naive,
        'a_true': a_true, 'b_true': b_true, 'dof': dof, 'N': N
    }

def plot_mc_results(results):
    """
    Create diagnostic plots for Monte Carlo results.
    """
    a_corr = results['a_corr']
    b_corr = results['b_corr']
    sigma_a_corr = results['sigma_a_corr']
    sigma_b_corr = results['sigma_b_corr']
    a_naive = results['a_naive']
    b_naive = results['b_naive']
    sigma_a_naive = results['sigma_a_naive']
    sigma_b_naive = results['sigma_b_naive']
    chi2_corr = results['chi2_corr']
    chi2_naive = results['chi2_naive']
    a_true = results['a_true']
    b_true = results['b_true']
    dof = results['dof']

    # ========== FIGURE 5: Parameter distributions ==========
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Slope a: histogram and statistics
    axes[0, 0].hist(a_corr, bins=15, alpha=0.6, color='C0', edgecolor='black', label='Correlated')
    axes[0, 0].hist(a_naive, bins=15, alpha=0.6, color='C1', edgecolor='black', label='Naive')
    axes[0, 0].axvline(a_true, color='k', linestyle='--', linewidth=2, label='True value')
    axes[0, 0].set_xlabel('Slope a', fontsize=11)
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].set_title('Distribution of Slope Estimates', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Intercept b: histogram and statistics
    axes[0, 1].hist(b_corr, bins=15, alpha=0.6, color='C0', edgecolor='black', label='Correlated')
    axes[0, 1].hist(b_naive, bins=15, alpha=0.6, color='C1', edgecolor='black', label='Naive')
    axes[0, 1].axvline(b_true, color='k', linestyle='--', linewidth=2, label='True value')
    axes[0, 1].set_xlabel('Intercept b', fontsize=11)
    axes[0, 1].set_ylabel('Count', fontsize=11)
    axes[0, 1].set_title('Distribution of Intercept Estimates', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Predicted vs observed uncertainties for a
    axes[1, 0].scatter(sigma_a_corr, a_corr - a_true, alpha=0.6, s=30, color='C0', label='Correlated')
    axes[1, 0].scatter(sigma_a_naive, a_naive - a_true, alpha=0.6, s=30, color='C1', label='Naive')
    axes[1, 0].axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    axes[1, 0].set_xlabel('Predicted uncertainty σ_a', fontsize=11)
    axes[1, 0].set_ylabel('Residual (a_fit - a_true)', fontsize=11)
    axes[1, 0].set_title('Slope: Predicted σ vs Actual Error', fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Predicted vs observed uncertainties for b
    axes[1, 1].scatter(sigma_b_corr, b_corr - b_true, alpha=0.6, s=30, color='C0', label='Correlated')
    axes[1, 1].scatter(sigma_b_naive, b_naive - b_true, alpha=0.6, s=30, color='C1', label='Naive')
    axes[1, 1].axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    axes[1, 1].set_xlabel('Predicted uncertainty σ_b', fontsize=11)
    axes[1, 1].set_ylabel('Residual (b_fit - b_true)', fontsize=11)
    axes[1, 1].set_title('Intercept: Predicted σ vs Actual Error', fontsize=12)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("/Users/vasilybelokurov/IoA Dropbox/Dr V.A. Belokurov/Code/correlated_noise/plots/fig_mc_distributions.png", dpi=150)
    print("\nSaved: plots/fig_mc_distributions.png")
    plt.close(fig)

    # ========== FIGURE 6: Chi-squared distributions ==========
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    chi2_dof_corr = chi2_corr / dof
    chi2_dof_naive = chi2_naive / dof

    axes[0].hist(chi2_dof_corr, bins=20, alpha=0.7, color='C0', edgecolor='black',
                label=f'Correlated (mean={chi2_dof_corr.mean():.3f})')
    axes[0].axvline(1.0, color='k', linestyle='--', linewidth=2, label='Expected (χ²/dof = 1)')
    axes[0].set_xlabel('χ²/dof', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Correlated Fit: χ²/dof Distribution', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].hist(chi2_dof_naive, bins=20, alpha=0.7, color='C1', edgecolor='black',
                label=f'Naive (mean={chi2_dof_naive.mean():.3f})')
    axes[1].axvline(1.0, color='k', linestyle='--', linewidth=2, label='Expected (χ²/dof = 1)')
    axes[1].set_xlabel('χ²/dof', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('Naive Fit: χ²/dof Distribution', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig("/Users/vasilybelokurov/IoA Dropbox/Dr V.A. Belokurov/Code/correlated_noise/plots/fig_mc_chisq.png", dpi=150)
    print("Saved: plots/fig_mc_chisq.png")
    plt.close(fig)

if __name__ == "__main__":
    results = monte_carlo_study(n_realizations=100)
    plot_mc_results(results)
