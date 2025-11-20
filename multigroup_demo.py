import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================
# Utilities to build block-correlated covariance
# ============================================================

def make_block_sizes(N, min_block=3, max_block=7, random_seed=123):
    """
    Partition N indices into blocks of size between min_block and max_block.
    The last block is adjusted if needed to hit N exactly.

    Returns:
      block_sizes: list of block lengths that sum to N
    """
    rng = np.random.default_rng(random_seed)
    remaining = N
    block_sizes = []

    while remaining > 0:
        if remaining <= max_block and remaining >= min_block:
            # last block fits in range
            block_sizes.append(remaining)
            remaining = 0
        elif remaining < min_block:
            # add what is left to last block
            if block_sizes:
                block_sizes[-1] += remaining
            else:
                block_sizes.append(remaining)
            remaining = 0
        else:
            size = rng.integers(min_block, max_block + 1)
            if remaining - size < 0:
                size = remaining
            block_sizes.append(size)
            remaining -= size

    assert sum(block_sizes) == N
    return block_sizes

def make_block_covariance(block_sizes, sigma, rho):
    """
    Build an N x N block-diagonal covariance matrix where within each block
    of size B the covariance is:

      C_ij = sigma^2 * [(1 - rho) * delta_ij + rho]

    Different blocks are uncorrelated (zeros in off-diagonal blocks).

    Args:
      block_sizes : list of ints, sizes of consecutive blocks, sum = N
      sigma       : standard deviation of each point
      rho         : correlation coefficient within each block (0 < rho < 1)

    Returns:
      C : (N, N) covariance matrix
      block_id : array of length N with integer block labels (0, 1, 2, ...)
    """
    N = sum(block_sizes)
    C = np.zeros((N, N), dtype=float)
    block_id = np.empty(N, dtype=int)

    start = 0
    bindex = 0
    for size in block_sizes:
        end = start + size
        # base block: sigma^2 on diag, rho*sigma^2 off-diag
        block = np.full((size, size), rho * sigma**2, dtype=float)
        np.fill_diagonal(block, sigma**2)
        C[start:end, start:end] = block
        block_id[start:end] = bindex
        start = end
        bindex += 1

    return C, block_id

# ============================================================
# Data generation
# ============================================================

def generate_mock_data_blocks(N=100, a_true=2.0, b_true=1.0,
                              x_min=0.0, x_max=10.0,
                              sigma=0.1, rho=0.8,
                              min_block=3, max_block=7,
                              random_seed=123):
    """
    Generate mock data for y = a_true * x + b_true with block-correlated noise.
    """
    rng = np.random.default_rng(random_seed)
    x = np.linspace(x_min, x_max, N)
    y_true = a_true * x + b_true

    block_sizes = make_block_sizes(N, min_block=min_block,
                                   max_block=max_block,
                                   random_seed=random_seed)
    C, block_id = make_block_covariance(block_sizes, sigma, rho)

    L = np.linalg.cholesky(C)
    z = rng.normal(size=N)
    noise = L @ z
    y_obs = y_true + noise

    return x, y_obs, y_true, C, block_id, block_sizes

# ============================================================
# Fitting routines
# ============================================================

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

    # Design matrix A: columns [x, 1]
    A = np.vstack([x, np.ones_like(x)]).T

    # Cholesky of C
    L = np.linalg.cholesky(C)

    # Work with transformed system: (L^{-1} A, L^{-1} y)
    A_tilde = np.linalg.solve(L, A)
    y_tilde = np.linalg.solve(L, y)

    ATA = A_tilde.T @ A_tilde
    ATy = A_tilde.T @ y_tilde

    cov_theta = np.linalg.inv(ATA)
    theta_hat = cov_theta @ ATy

    # chi^2 = (y - A theta)^T C^{-1} (y - A theta)
    residuals = y - A @ theta_hat
    r_tilde = np.linalg.solve(L, residuals)
    chi2 = r_tilde @ r_tilde

    return theta_hat, cov_theta, chi2

def fit_line_naive(x, y, sigma_eff):
    """
    Naive fit: assume independent Gaussian errors with
    variance sigma_eff^2 for all points.

    sigma_eff can be:
      - scalar: same sigma for all points
      - array length N: per-point sigma
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

    # Weighted least squares with diagonal covariance:
    # A_w = A / sigma, y_w = y / sigma
    A_w = A / sigma_vec[:, None]
    y_w = y / sigma_vec

    ATA = A_w.T @ A_w
    ATy = A_w.T @ y_w

    cov_theta = np.linalg.inv(ATA)
    theta_hat = cov_theta @ ATy

    residuals = y - A @ theta_hat
    chi2 = np.sum((residuals / sigma_vec)**2)

    return theta_hat, cov_theta, chi2

# ============================================================
# Main demo
# ============================================================

def main():
    # True parameters and noise properties
    a_true = 2.0
    b_true = 1.0
    sigma_noise = 0.15   # per-point sigma
    rho = 0.7            # within-block correlation
    N = 100

    # Generate one mock data set with block-correlated noise
    x, y_obs, y_true, C, block_id, block_sizes = generate_mock_data_blocks(
        N=N,
        a_true=a_true,
        b_true=b_true,
        sigma=sigma_noise,
        rho=rho,
        min_block=3,
        max_block=7,
        random_seed=123,
    )

    print("="*70)
    print("MULTI-GROUP DATASET WITH BLOCK-CORRELATED NOISE")
    print("="*70)
    print("\nBlock sizes (should sum to {}):".format(N))
    print(block_sizes)
    print("Number of blocks: {}".format(len(block_sizes)))
    print("Within-block correlation: ρ = {:.2f}".format(rho))
    print()

    # Correct fit with full covariance
    theta_corr, cov_corr, chi2_corr = fit_line_correlated(x, y_obs, C)
    a_corr, b_corr = theta_corr
    sigma_a_corr = np.sqrt(cov_corr[0, 0])
    sigma_b_corr = np.sqrt(cov_corr[1, 1])

    # Naive fit: treat data as independent with an effective sigma
    sigma_diag = np.sqrt(np.diag(C))
    sigma_eff = np.median(sigma_diag)

    theta_naive, cov_naive, chi2_naive = fit_line_naive(x, y_obs, sigma_eff)
    a_naive, b_naive = theta_naive
    sigma_a_naive = np.sqrt(cov_naive[0, 0])
    sigma_b_naive = np.sqrt(cov_naive[1, 1])

    dof = N - 2

    print("True parameters:")
    print("  a_true = {:.4f}, b_true = {:.4f}".format(a_true, b_true))
    print()

    print("Correct fit (full block-diagonal covariance):")
    print("  a = {:.4f} +/- {:.4f}".format(a_corr, sigma_a_corr))
    print("  b = {:.4f} +/- {:.4f}".format(b_corr, sigma_b_corr))
    print("  χ²/dof = {:.2f} / {} = {:.3f}".format(chi2_corr, dof, chi2_corr / dof))
    print()

    print("Naive fit (independent errors, σ_eff = {:.4f}):".format(sigma_eff))
    print("  a = {:.4f} +/- {:.4f}".format(a_naive, sigma_a_naive))
    print("  b = {:.4f} +/- {:.4f}".format(b_naive, sigma_b_naive))
    print("  χ²/dof = {:.2f} / {} = {:.3f}".format(chi2_naive, dof, chi2_naive / dof))
    print()

    print("Uncertainty reduction (naive/correct):")
    print("  σ_a: {:.2f}× smaller".format(sigma_a_corr / sigma_a_naive))
    print("  σ_b: {:.2f}× smaller".format(sigma_b_corr / sigma_b_naive))
    print("="*70)

    # ========== FIGURE 1: Data and fits with block coloring ==========
    fig, ax = plt.subplots(figsize=(14, 6))

    # Color scheme for blocks
    colors = plt.cm.tab20(np.linspace(0, 1, len(block_sizes)))

    # Plot data colored by block
    for bid in range(len(block_sizes)):
        mask = block_id == bid
        ax.scatter(x[mask], y_obs[mask], s=50, alpha=0.7, color=colors[bid],
                  edgecolor='black', linewidth=0.5, label=f'Block {bid}' if bid < 8 else '')

    # Plot fits
    x_fine = np.linspace(x.min(), x.max(), 500)
    y_fit_corr = a_corr * x_fine + b_corr
    y_fit_naive = a_naive * x_fine + b_naive

    ax.plot(x_fine, a_true * x_fine + b_true, 'k--', linewidth=2.5, label='True line', zorder=5)
    ax.plot(x_fine, y_fit_corr, 'C0-', linewidth=2, label='Fit (correlated)', zorder=4)
    ax.plot(x_fine, y_fit_naive, 'C1-', linewidth=2, label='Fit (naive)', zorder=4)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Linear Fit with Block-Correlated Noise (N={N}, blocks={len(block_sizes)}, ρ={rho:.1f})',
                 fontsize=13)
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("/Users/vasilybelokurov/IoA Dropbox/Dr V.A. Belokurov/Code/correlated_noise/plots/fig_multigroup_fits.png", dpi=150)
    print("\nSaved: plots/fig_multigroup_fits.png")
    plt.close(fig)

    # ========== FIGURE 2: Covariance and correlation matrices with block structure ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Covariance matrix
    im1 = axes[0].imshow(C, cmap='RdBu_r', aspect='auto')
    axes[0].set_xlabel('Point index i', fontsize=11)
    axes[0].set_ylabel('Point index j', fontsize=11)
    axes[0].set_title('Block-Diagonal Covariance Matrix C', fontsize=12)
    plt.colorbar(im1, ax=axes[0], label='Covariance')

    # Add block boundaries
    block_ends = np.cumsum(block_sizes)
    for end in block_ends[:-1]:
        axes[0].axhline(end - 0.5, color='lime', linewidth=1.5, alpha=0.7)
        axes[0].axvline(end - 0.5, color='lime', linewidth=1.5, alpha=0.7)

    # Correlation matrix
    corr_matrix = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    im2 = axes[1].imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes[1].set_xlabel('Point index i', fontsize=11)
    axes[1].set_ylabel('Point index j', fontsize=11)
    axes[1].set_title('Correlation Matrix ρ_ij', fontsize=12)
    plt.colorbar(im2, ax=axes[1], label='Correlation')

    # Add block boundaries
    for end in block_ends[:-1]:
        axes[1].axhline(end - 0.5, color='lime', linewidth=1.5, alpha=0.7)
        axes[1].axvline(end - 0.5, color='lime', linewidth=1.5, alpha=0.7)

    fig.tight_layout()
    fig.savefig("/Users/vasilybelokurov/IoA Dropbox/Dr V.A. Belokurov/Code/correlated_noise/plots/fig_multigroup_covariance.png", dpi=150)
    print("Saved: plots/fig_multigroup_covariance.png")
    plt.close(fig)

    # ========== FIGURE 3: Residuals comparison ==========
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    residuals_corr = y_obs - (a_corr * x + b_corr)
    residuals_naive = y_obs - (a_naive * x + b_naive)

    # Correlated fit residuals with block coloring
    for bid in range(len(block_sizes)):
        mask = block_id == bid
        axes[0].scatter(x[mask], residuals_corr[mask], s=50, alpha=0.7, color=colors[bid],
                       edgecolor='black', linewidth=0.5)

    axes[0].axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=1.5)
    axes[0].set_ylabel('Residuals', fontsize=11)
    axes[0].set_title('Correlated Fit Residuals', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Add block boundaries
    for end in block_ends[:-1]:
        axes[0].axvline(x[end-1], color='gray', linewidth=1, alpha=0.3, linestyle=':')

    # Naive fit residuals with block coloring
    for bid in range(len(block_sizes)):
        mask = block_id == bid
        axes[1].scatter(x[mask], residuals_naive[mask], s=50, alpha=0.7, color=colors[bid],
                       edgecolor='black', linewidth=0.5)

    axes[1].axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=1.5)
    axes[1].set_xlabel('x', fontsize=11)
    axes[1].set_ylabel('Residuals', fontsize=11)
    axes[1].set_title('Naive Fit Residuals', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Add block boundaries
    for end in block_ends[:-1]:
        axes[1].axvline(x[end-1], color='gray', linewidth=1, alpha=0.3, linestyle=':')

    fig.tight_layout()
    fig.savefig("/Users/vasilybelokurov/IoA Dropbox/Dr V.A. Belokurov/Code/correlated_noise/plots/fig_multigroup_residuals.png", dpi=150)
    print("Saved: plots/fig_multigroup_residuals.png")
    plt.close(fig)

    # ========== FIGURE 4: Chi-squared comparison ==========
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['Correlated', 'Naive']
    chi2_dof_values = [chi2_corr/dof, chi2_naive/dof]
    colors_bar = ['C0', 'C1']

    x_pos = np.arange(len(methods))
    width = 0.35

    bars = ax.bar(x_pos, chi2_dof_values, width, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(1.0, color='k', linestyle='--', linewidth=2.5, label='Expected (χ²/dof = 1)')

    # Add value labels on bars
    for bar, val in zip(bars, chi2_dof_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('χ²/dof', fontsize=12)
    ax.set_title(f'Goodness of Fit: N={N} points, {len(block_sizes)} blocks', fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim([0, max(chi2_dof_values) * 1.4])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig("/Users/vasilybelokurov/IoA Dropbox/Dr V.A. Belokurov/Code/correlated_noise/plots/fig_multigroup_chisq.png", dpi=150)
    print("Saved: plots/fig_multigroup_chisq.png")
    plt.close(fig)

    # ========== FIGURE 5: Parameter comparison table ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    table_data = [
        ['Parameter', 'True', 'Corr. Fit', 'Naive Fit', 'Corr. σ', 'Naive σ'],
        ['a (slope)', f'{a_true:.4f}', f'{a_corr:.4f}', f'{a_naive:.4f}',
         f'{sigma_a_corr:.4f}', f'{sigma_a_naive:.4f}'],
        ['b (intercept)', f'{b_true:.4f}', f'{b_corr:.4f}', f'{b_naive:.4f}',
         f'{sigma_b_corr:.4f}', f'{sigma_b_naive:.4f}'],
        ['χ²/dof', '-', f'{chi2_corr/dof:.3f}', f'{chi2_naive/dof:.3f}', '-', '-'],
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.15, 0.12, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Color header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax.set_title(f'Parameter Estimates (N={N} points, {len(block_sizes)} blocks, ρ={rho:.2f})',
                fontsize=13, fontweight='bold', pad=20)

    fig.tight_layout()
    fig.savefig("/Users/vasilybelokurov/IoA Dropbox/Dr V.A. Belokurov/Code/correlated_noise/plots/fig_multigroup_table.png", dpi=150)
    print("Saved: plots/fig_multigroup_table.png")
    plt.close(fig)

if __name__ == "__main__":
    main()
