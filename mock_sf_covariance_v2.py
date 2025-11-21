"""
Mock SF covariance experiment with epoch and block bootstrap methods.

This script:
1. Simulates many DRW lightcurves to estimate "true" SF covariance
2. Picks one realization and applies:
   - Naive epoch bootstrap
   - Block bootstrap
3. Compares both to the ground truth

Version 2: Includes block bootstrap implementation.
"""
import numpy as np


def generate_irregular_times(n_epochs, t_max, rng):
    """
    Generate an irregular time grid between 0 and t_max.

    Parameters
    ----------
    n_epochs : int
        Number of epochs.
    t_max : float
        Maximum time (e.g. years).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    times : ndarray, shape (n_epochs,)
        Sorted array of observation times.
    """
    # Start from uniform and add small jitter
    t = np.sort(rng.uniform(0.0, t_max, size=n_epochs))
    jitter = rng.normal(scale=0.05 * (t_max / n_epochs), size=n_epochs)
    t = np.sort(t + jitter)
    # Ensure within [0, t_max]
    t = np.clip(t, 0.0, t_max)
    return t


def ou_covariance(times, sigma_drw, tau_drw):
    """
    OU (damped random walk) covariance matrix for given times.

    C(dt) = sigma_drw^2 * exp( -|dt| / tau_drw )

    Parameters
    ----------
    times : ndarray, shape (n_epochs,)
    sigma_drw : float
        DRW amplitude in magnitudes.
    tau_drw : float
        DRW damping timescale (same units as times).

    Returns
    -------
    C : ndarray, shape (n_epochs, n_epochs)
        Covariance matrix.
    """
    t = times[:, None]
    dt = np.abs(t - t.T)
    return sigma_drw**2 * np.exp(-dt / tau_drw)


def simulate_drw_lightcurves(times, sigma_drw, tau_drw,
                             n_realizations, rng, sigma_phot=0.0):
    """
    Simulate n_realizations DRW lightcurves on the same time grid.

    Parameters
    ----------
    times : ndarray, shape (n_epochs,)
    sigma_drw : float
    tau_drw : float
    n_realizations : int
    rng : np.random.Generator
    sigma_phot : float, optional
        Gaussian photometric noise per point (mag).

    Returns
    -------
    mags : ndarray, shape (n_realizations, n_epochs)
        Simulated lightcurves.
    """
    n_epochs = len(times)
    C = ou_covariance(times, sigma_drw, tau_drw)
    # Add small nugget for numerical stability
    C += 1e-10 * np.eye(n_epochs)
    L = np.linalg.cholesky(C)
    # Draw lightcurves
    z = rng.normal(size=(n_realizations, n_epochs))
    m = z @ L.T  # shape (n_realizations, n_epochs)
    if sigma_phot > 0.0:
        m += rng.normal(scale=sigma_phot, size=m.shape)
    return m


def compute_sf_vector(times, mags, lags, lag_width,
                      z=0.0, sigma_phot=0.0):
    """
    Compute SF^2 at specified rest-frame lags for one lightcurve.

    Uses pairwise magnitude differences and log-lag binning.

    Parameters
    ----------
    times : ndarray, shape (n_epochs,)
    mags : ndarray, shape (n_epochs,)
    lags : ndarray, shape (n_lags,)
        Rest-frame lag centers.
    lag_width : float
        Half-width in log10-space (dex) for the lag bins.
    z : float
        Redshift (for rest-frame lag).
    sigma_phot : float
        Per-epoch photometric noise (mag). Used to subtract noise variance.

    Returns
    -------
    sf2 : ndarray, shape (n_lags,)
        SF^2 estimates at the chosen lags.
    """
    times = np.asarray(times)
    mags = np.asarray(mags)
    n_epochs = len(times)

    # All pairwise indices (upper triangle)
    idx_i, idx_j = np.triu_indices(n_epochs, k=1)
    dt_obs = np.abs(times[idx_j] - times[idx_i])
    dt_rf = dt_obs / (1.0 + z)
    dm = mags[idx_j] - mags[idx_i]

    # Remove zero-lag pairs (can arise after bootstrap resampling)
    nz = dt_rf > 0.0
    dt_rf = dt_rf[nz]
    dm = dm[nz]

    # Noise variance per pair (if constant per-epoch noise)
    sigma_pair2 = 2.0 * sigma_phot**2

    # Precompute log10 lags of pairs
    log_dt = np.log10(dt_rf)

    sf2 = np.zeros_like(lags, dtype=float)
    for k, tau in enumerate(lags):
        log_tau = np.log10(tau)
        mask = np.abs(log_dt - log_tau) <= lag_width
        if not np.any(mask):
            sf2[k] = np.nan
            continue
        dm2 = dm[mask]**2
        # Noise-corrected SF^2
        sf2_k = np.mean(dm2 - sigma_pair2)
        # Clip to positive small value to avoid log10 of non-positive
        sf2[k] = max(sf2_k, 1e-8)
    return sf2


def compute_A_vector(times, mags, lags, lag_width,
                     z=0.0, sigma_phot=0.0):
    """
    Compute A_k = log10(SF_k^2) at the chosen lags.

    Handles NaNs by filling with the smallest finite SF^2 in the vector.
    """
    sf2 = compute_sf_vector(times, mags, lags, lag_width,
                            z=z, sigma_phot=sigma_phot)
    if np.any(np.isnan(sf2)):
        valid = np.isfinite(sf2)
        if not np.any(valid):
            raise RuntimeError("All SF^2 values are NaN for this lightcurve.")
        min_val = np.nanmin(sf2[valid])
        sf2[~valid] = min_val
    A = np.log10(sf2)
    return A


def estimate_true_covariance(times, all_mags, lags, lag_width,
                             z=0.0, sigma_phot=0.0):
    """
    Estimate the "true" covariance of the A-vector from many lightcurves.

    Parameters
    ----------
    times : ndarray, shape (n_epochs,)
    all_mags : ndarray, shape (n_realizations, n_epochs)
        Simulated lightcurves.
    lags : ndarray, shape (n_lags,)
    lag_width : float
    z : float
    sigma_phot : float

    Returns
    -------
    cov_true : ndarray, shape (n_lags, n_lags)
        Monte Carlo covariance estimate of A.
    mean_A : ndarray, shape (n_lags,)
        Mean A over realizations.
    A_mat : ndarray, shape (n_realizations, n_lags)
        All A-vectors.
    """
    A_list = []
    for mags in all_mags:
        A = compute_A_vector(times, mags, lags, lag_width,
                             z=z, sigma_phot=sigma_phot)
        A_list.append(A)
    A_mat = np.vstack(A_list)
    mean_A = A_mat.mean(axis=0)
    X = A_mat - mean_A
    cov_true = X.T @ X / (A_mat.shape[0] - 1)
    return cov_true, mean_A, A_mat


def bootstrap_sf_covariance(times, mags, lags, lag_width,
                            n_bootstrap, rng, z=0.0, sigma_phot=0.0):
    """
    Epoch bootstrap: resample individual epochs with replacement.

    This is the naive bootstrap that is known to be problematic for
    time series SFs.

    Parameters
    ----------
    times : ndarray, shape (n_epochs,)
    mags : ndarray, shape (n_epochs,)
    lags : ndarray, shape (n_lags,)
    lag_width : float
    n_bootstrap : int
    rng : np.random.Generator
    z : float
    sigma_phot : float

    Returns
    -------
    cov_boot : ndarray, shape (n_lags, n_lags)
        Bootstrap covariance estimate.
    mean_A_boot : ndarray, shape (n_lags,)
        Mean A over bootstrap samples.
    A_boot : ndarray, shape (n_bootstrap, n_lags)
        All bootstrap A-vectors.
    """
    n_epochs = len(times)
    A_boot = []
    for _ in range(n_bootstrap):
        # Resample epochs with replacement
        idx = rng.integers(0, n_epochs, size=n_epochs)
        t_b = times[idx]
        m_b = mags[idx]
        # Sort by time to keep a lightcurve-like structure
        order = np.argsort(t_b)
        t_b = t_b[order]
        m_b = m_b[order]
        A_b = compute_A_vector(t_b, m_b, lags, lag_width,
                               z=z, sigma_phot=sigma_phot)
        A_boot.append(A_b)
    A_boot = np.vstack(A_boot)
    mean_A_boot = A_boot.mean(axis=0)
    X = A_boot - mean_A_boot
    cov_boot = X.T @ X / (A_boot.shape[0] - 1)
    return cov_boot, mean_A_boot, A_boot


def block_bootstrap_timeseries(times, mags, block_size, rng):
    """
    Circular block bootstrap for a single lightcurve.

    Parameters
    ----------
    times : ndarray, shape (n_epochs,)
    mags : ndarray, shape (n_epochs,)
    block_size : int
        Length of each block in epochs.
    rng : np.random.Generator

    Returns
    -------
    t_b, m_b : ndarrays, shape (n_epochs,)
        Bootstrap resampled time series (sorted by time).
    """
    n = len(times)
    if block_size <= 0 or block_size > n:
        raise ValueError("block_size must be in [1, n_epochs]")

    # Number of blocks needed to reach length >= n
    n_blocks = int(np.ceil(n / block_size))

    idx_list = []
    for _ in range(n_blocks):
        # Random starting index for circular block
        s = rng.integers(0, n)
        block_idx = (s + np.arange(block_size)) % n
        idx_list.append(block_idx)

    idx = np.concatenate(idx_list)[:n]
    t_b = times[idx]
    m_b = mags[idx]
    # Sort to impose increasing time, as in a real lightcurve
    order = np.argsort(t_b)
    return t_b[order], m_b[order]


def bootstrap_sf_covariance_block(times, mags, lags, lag_width,
                                  n_bootstrap, rng, z=0.0,
                                  sigma_phot=0.0, block_size=10):
    """
    Block bootstrap: resample blocks of consecutive epochs with replacement.

    This preserves local time correlations within blocks and is usually a
    better approximation for time series statistics than naive epoch bootstrap.

    Parameters
    ----------
    times : ndarray, shape (n_epochs,)
    mags : ndarray, shape (n_epochs,)
    lags : ndarray, shape (n_lags,)
    lag_width : float
    n_bootstrap : int
    rng : np.random.Generator
    z : float
    sigma_phot : float
    block_size : int
        Length of each block in epochs.

    Returns
    -------
    cov_boot : ndarray, shape (n_lags, n_lags)
        Bootstrap covariance estimate.
    mean_A_boot : ndarray, shape (n_lags,)
        Mean A over bootstrap samples.
    A_boot : ndarray, shape (n_bootstrap, n_lags)
        All bootstrap A-vectors.
    """
    A_boot = []
    for _ in range(n_bootstrap):
        t_b, m_b = block_bootstrap_timeseries(times, mags, block_size, rng)
        A_b = compute_A_vector(t_b, m_b, lags, lag_width,
                               z=z, sigma_phot=sigma_phot)
        A_boot.append(A_b)
    A_boot = np.vstack(A_boot)
    mean_A_boot = A_boot.mean(axis=0)
    X = A_boot - mean_A_boot
    cov_boot = X.T @ X / (A_boot.shape[0] - 1)
    return cov_boot, mean_A_boot, A_boot


def mock_experiment():
    """
    Run a complete mock experiment:

    - generate an irregular time grid
    - simulate many DRW lightcurves
    - estimate "true" SF covariance from many realizations
    - bootstrap SF covariance (epoch and block) from a single lightcurve
    - return both covariances and diagnostics
    """
    rng = np.random.default_rng(42)

    # Time sampling
    n_epochs = 80
    t_max = 5.0  # years
    times = generate_irregular_times(n_epochs, t_max, rng)

    # DRW parameters
    sigma_drw = 0.2   # mag
    tau_drw = 0.5     # years
    sigma_phot = 0.01  # mag

    # Lags (rest-frame, assume z = 0 here)
    n_lags = 8
    tau_min = 0.02  # years
    tau_max = 2.0   # years
    lags = np.logspace(np.log10(tau_min), np.log10(tau_max), n_lags)
    lag_width = 0.15  # dex half-width

    # Simulate many lightcurves to estimate "true" covariance
    n_realizations = 1000
    all_mags = simulate_drw_lightcurves(
        times, sigma_drw, tau_drw, n_realizations, rng, sigma_phot=sigma_phot
    )
    cov_true, mean_A_true, A_mat = estimate_true_covariance(
        times, all_mags, lags, lag_width, z=0.0, sigma_phot=sigma_phot
    )

    # Choose one realization as the "data" lightcurve for bootstrap
    mags_data = all_mags[0]
    n_bootstrap = 500

    # Naive epoch bootstrap
    cov_boot_epoch, mean_A_epoch, A_boot_epoch = bootstrap_sf_covariance(
        times, mags_data, lags, lag_width, n_bootstrap, rng,
        z=0.0, sigma_phot=sigma_phot
    )

    # Block bootstrap
    block_size = 10  # in epochs; tune as you like
    cov_boot_block, mean_A_block, A_boot_block = bootstrap_sf_covariance_block(
        times, mags_data, lags, lag_width, n_bootstrap, rng,
        z=0.0, sigma_phot=sigma_phot, block_size=block_size
    )

    # Diagnostics: mean absolute relative error in covariance
    def mean_abs_rel_error(cov_est, cov_true):
        rel = (cov_est - cov_true) / cov_true
        # Avoid division by zero for elements where cov_true == 0
        finite = np.isfinite(rel)
        return np.nanmean(np.abs(rel[finite]))

    mean_rel_err_epoch = mean_abs_rel_error(cov_boot_epoch, cov_true)
    mean_rel_err_block = mean_abs_rel_error(cov_boot_block, cov_true)

    results = {
        "times": times,
        "lags": lags,
        "cov_true": cov_true,
        "cov_boot_epoch": cov_boot_epoch,
        "cov_boot_block": cov_boot_block,
        "mean_rel_err_epoch": mean_rel_err_epoch,
        "mean_rel_err_block": mean_rel_err_block,
        "mean_A_true": mean_A_true,
        "mean_A_epoch": mean_A_epoch,
        "mean_A_block": mean_A_block,
        "block_size": block_size,
    }
    return results


if __name__ == "__main__":
    res = mock_experiment()
    print("=" * 70)
    print("BLOCK BOOTSTRAP VALIDATION RESULTS")
    print("=" * 70)
    print(f"\nLags (rest-frame):          {res['lags']}")
    print(f"\nMean abs rel error (epoch): {res['mean_rel_err_epoch']:.4f}")
    print(f"Mean abs rel error (block): {res['mean_rel_err_block']:.4f}")
    print(f"Improvement factor:         {res['mean_rel_err_epoch'] / res['mean_rel_err_block']:.2f}×")
    print(f"\nBlock size: {res['block_size']} epochs")
    print("=" * 70)
