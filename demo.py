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

    Returns:
        theta_hat  : array([a_hat, b_hat])
        cov_theta  : 2x2 covariance matrix of theta_hat
        chi2       : chi^2 at the best fit
    """
    x = np.asarray(x)
    y = np.asarray(y)
    N = len(x)

    # Design matrix A: columns [x, 1]
    A = np.vstack([x, np.ones_like(x)]).T

    # Cholesky of C
    L = np.linalg.cholesky(C)

    # Work with transformed system: (L^{-1} A, L^{-1} y)
    # to avoid explicitly forming C^{-1}.
    A_tilde = np.linalg.solve(L, A)
    y_tilde = np.linalg.solve(L, y)

    # Normal equations for weighted least squares
    ATA = A_tilde.T @ A_tilde
    ATy = A_tilde.T @ y_tilde

    cov_theta = np.linalg.inv(ATA)
    theta_hat = cov_theta @ ATy

    # Compute chi^2 = (y - A theta)^T C^{-1} (y - A theta)
    residuals = y - A @ theta_hat
    r_tilde = np.linalg.solve(L, residuals)
    chi2 = r_tilde @ r_tilde

    return theta_hat, cov_theta, chi2

def fit_line_naive(x, y, sigma_eff):
    """
    Naive fit: assume independent Gaussian errors with
    variance sigma_eff^2 for all points (or a vector of variances).

    sigma_eff can be:
      - scalar: same sigma for all points
      - array length N: per-point sigma
    """
    x = np.asarray(x)
    y = np.asarray(y)
    N = len(x)

    # Design matrix
    A = np.vstack([x, np.ones_like(x)]).T

    sigma_eff = np.asarray(sigma_eff)
    if sigma_eff.ndim == 0:
        sigma_vec = np.full(N, float(sigma_eff))
    else:
        assert len(sigma_eff) == N
        sigma_vec = sigma_eff

    # Weighted least squares with diagonal covariance:
    # define A_w = A / sigma, y_w = y / sigma
    A_w = A / sigma_vec[:, None]
    y_w = y / sigma_vec

    ATA = A_w.T @ A_w
    ATy = A_w.T @ y_w

    cov_theta = np.linalg.inv(ATA)
    theta_hat = cov_theta @ ATy

    # chi^2 with naive diagonal covariance
    residuals = y - A @ theta_hat
    chi2 = np.sum((residuals / sigma_vec)**2)

    return theta_hat, cov_theta, chi2

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

    # Naive fit: treat data as independent with variance from diag(C)
    sigma_diag = np.sqrt(np.diag(C))
    # One simple choice: use the same per-point sigma = median of sigma_diag
    sigma_eff = np.median(sigma_diag)

    theta_naive, cov_naive, chi2_naive = fit_line_naive(x, y_obs, sigma_eff)
    a_naive, b_naive = theta_naive
    sigma_a_naive = np.sqrt(cov_naive[0, 0])
    sigma_b_naive = np.sqrt(cov_naive[1, 1])

    dof = N - 2

    print("True parameters:")
    print("  a_true = {:.4f}, b_true = {:.4f}".format(a_true, b_true))
    print()

    print("Correct correlated fit:")
    print("  a = {:.4f} +/- {:.4f}".format(a_corr, sigma_a_corr))
    print("  b = {:.4f} +/- {:.4f}".format(b_corr, sigma_b_corr))
    print("  chi2 / dof = {:.2f} / {} = {:.2f}".format(chi2_corr, dof, chi2_corr / dof))
    print()

    print("Naive uncorrelated fit:")
    print("  a = {:.4f} +/- {:.4f}".format(a_naive, sigma_a_naive))
    print("  b = {:.4f} +/- {:.4f}".format(b_naive, sigma_b_naive))
    print("  chi2 / dof = {:.2f} / {} = {:.2f}".format(chi2_naive, dof, chi2_naive / dof))
    print()

    # Plot the data and fits for visual comparison
    x_fine = np.linspace(x.min(), x.max(), 400)
    y_fit_corr = a_corr * x_fine + b_corr
    y_fit_naive = a_naive * x_fine + b_naive

    plt.figure()
    plt.errorbar(x, y_obs, yerr=np.sqrt(np.diag(C)), fmt="o", label="data", alpha=0.7)
    plt.plot(x_fine, a_true * x_fine + b_true, label="true line", linestyle="--")
    plt.plot(x_fine, y_fit_corr, label="fit (correlated)")
    plt.plot(x_fine, y_fit_naive, label="fit (naive)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/Users/vasilybelokurov/IoA Dropbox/Dr V.A. Belokurov/Code/correlated_noise/fig_fits.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
