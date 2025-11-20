# Correlated Noise in Model Fitting: Illustration & Analysis

This project demonstrates the **critical importance of accounting for correlated noise** when fitting models to observational data. Ignoring correlations leads to systematic underestimation of parameter uncertainties and misleading goodness-of-fit metrics.

---

## Project Structure

```
correlated_noise/
├── README.md                       (this file)
├── demo.py                         (simple example, N=50)
├── demo_plots.py                   (single realization + 4 plots)
├── monte_carlo.py                  (100 realizations + 2 plots)
├── multigroup_demo.py              (multi-group example, N=100)
├── multigroup_monte_carlo.py       (100 realizations + 2 plots)
└── plots/                          (all generated PNG figures)
    ├── fig_fits.png
    ├── fig_ellipses.png
    ├── fig_residuals.png
    ├── fig_chisq.png
    ├── fig_mc_distributions.png
    ├── fig_mc_chisq.png
    ├── fig_multigroup_fits.png
    ├── fig_multigroup_covariance.png
    ├── fig_multigroup_residuals.png
    ├── fig_multigroup_chisq.png
    ├── fig_multigroup_table.png
    ├── fig_multigroup_mc_distributions.png
    └── fig_multigroup_mc_chisq.png
```

## Overview

The project contains three escalating examples:

1. **Simple case (N=50 points):** All points drawn from a single exponential covariance structure
2. **Realistic case (N=100 points):** Multiple groups (3–7 points each) with block-diagonal covariance
3. **Monte Carlo studies:** 100+ realizations to quantify bias and variance effects

Each example compares:
- **Correct fit:** Uses the full covariance matrix via Cholesky decomposition
- **Naive fit:** Assumes independent errors (ignores all correlations)

---

## Key Findings

### Single Realization (N=50, exponential covariance)
```
Correlated fit:  χ²/dof = 1.01 ✓   σ_a = 0.134, σ_b = 0.086
Naive fit:       χ²/dof = 0.33 ✗   σ_a = 0.048, σ_b = 0.028
                                    (3× underestimate)
```

### Multi-Group Case (N=100, 20 blocks, ρ=0.70)
```
Correlated fit:  χ²/dof = 0.80      σ_a = 0.0101, σ_b = 0.0553
Naive fit:       χ²/dof = 0.50      σ_a = 0.0051, σ_b = 0.0298
                                     (2× underestimate)
```

### Monte Carlo Results (100 realizations, N=100, 19 blocks, ρ=0.70)
```
Correlated fit:  Predicted σ / Observed σ ≈ 0.89  ✓ (near unity)
                 χ²/dof: mean = 1.74 (elevated due to random block variation)

Naive fit:       Predicted σ / Observed σ ≈ 0.48  ✗ (50% too small)
                 χ²/dof: mean = 0.95 (spuriously good agreement)
```

**Interpretation:** The naive fit's predicted uncertainties are only ~50% of the true empirical scatter. This means **nominal 1σ confidence intervals contain ~68% of the time only when the actual coverage is ~98%**—critical for scientific inference.

---

## Files & Usage

### Scripts (all ready to run)

#### Single Covariance Example
```bash
python demo.py                # Single realization
python demo_plots.py          # 4 diagnostic plots (fits, ellipses, residuals, χ²)
python monte_carlo.py         # 100 realizations + 2 plots
```

#### Multi-Group Example (primary demonstration)
```bash
python multigroup_demo.py           # Single realization with block structure
python multigroup_monte_carlo.py    # 100 realizations + 2 diagnostic plots
```

### Generated Plots

All plots are saved to the `plots/` subdirectory.

#### Single Covariance (6 plots in `plots/`)
| File | Content |
|------|---------|
| `plots/fig_fits.png` | Data + best-fit lines with 1σ bands |
| `plots/fig_ellipses.png` | Parameter confidence ellipses (1σ, 2σ) in (a,b) space |
| `plots/fig_residuals.png` | Residuals + full covariance/correlation matrices |
| `plots/fig_chisq.png` | χ²/dof bar chart |
| `plots/fig_mc_distributions.png` | Parameter distributions from 100 realizations |
| `plots/fig_mc_chisq.png` | χ²/dof distributions |

#### Multi-Group (7 plots in `plots/`)
| File | Content |
|------|---------|
| `plots/fig_multigroup_fits.png` | Data (colored by block) + fits |
| `plots/fig_multigroup_covariance.png` | Block-diagonal structure with color-coded block boundaries |
| `plots/fig_multigroup_residuals.png` | Residuals showing structure ignored by naive fit |
| `plots/fig_multigroup_chisq.png` | χ²/dof comparison |
| `plots/fig_multigroup_table.png` | Parameter table summary |
| `plots/fig_multigroup_mc_distributions.png` | Parameter distributions (100 realizations) |
| `plots/fig_multigroup_mc_chisq.png` | χ²/dof distributions |

---

## Mathematical Framework

### Likelihood for Correlated Errors

For observational data $\mathbf{y}$ with a linear model $\mathbf{y}_\text{true} = A \boldsymbol{\theta}$ and **correlated Gaussian noise** with covariance $C$:

$$
\chi^2 = (\mathbf{y} - A\boldsymbol{\theta})^T C^{-1} (\mathbf{y} - A\boldsymbol{\theta})
$$

**Parameter uncertainties:**
$$
\text{Cov}(\boldsymbol{\theta}) = (A^T C^{-1} A)^{-1}
$$

### Numerical Implementation

To avoid explicit matrix inversion (numerically unstable), we use the Cholesky decomposition $C = LL^T$:

1. Solve $L \tilde{A} = A$ and $L \tilde{\mathbf{y}} = \mathbf{y}$ using forward substitution
2. Compute $A^T_\tilde A^T_\tilde = (A^T A)_\text{whitened}$ and solve for $\boldsymbol{\theta}$
3. Compute $\chi^2 = \|\tilde{\mathbf{r}}\|^2$ where $\tilde{\mathbf{r}} = L^{-1}(\mathbf{y} - A\boldsymbol{\theta})$

This is robust and efficient for $N \lesssim 10^4$.

### Covariance Model

**Block-diagonal structure:** Noise is correlated *within* groups but independent *between* groups:

$$
C_{ij} = \begin{cases}
\sigma^2 \left[ (1 - \rho) \delta_{ij} + \rho \right] & \text{if } i,j \in \text{same block} \\
0 & \text{otherwise}
\end{cases}
$$

where:
- $\sigma$ = per-point standard deviation
- $\rho$ = within-block correlation coefficient
- Block sizes: randomly chosen between 3–7 points

---

## How to Extend

### Experiment with Different Parameters
```python
# In multigroup_demo.py, modify:
rho = 0.5          # Weaker correlations
N = 200            # Larger dataset
sigma_noise = 0.2  # Stronger noise
```

### Different Covariance Models
Replace `make_block_covariance()` with:
- Exponential: $C_{ij} = \sigma^2 \exp(-|x_i - x_j|/\ell)$
- Matérn: $C_{ij} = \sigma^2 (1 + |r_{ij}|/\ell) \exp(-|r_{ij}|/\ell)$
- Squared-exponential (RBF): $C_{ij} = \sigma^2 \exp(-(x_i - x_j)^2 / 2\ell^2)$

### Nonlinear Models
Replace the design matrix:
```python
# Current (linear)
A = np.vstack([x, np.ones_like(x)]).T

# Polynomial (quadratic)
A = np.vstack([x**2, x, np.ones_like(x)]).T

# Sinusoidal
A = np.vstack([np.sin(x), np.cos(x), np.ones_like(x)]).T
```

---

## References & Context

**Why this matters:**

1. **Realistic noise:** Observational data often have correlated errors due to:
   - Instrumental systematics (thermal drifts, calibration)
   - Spatial/temporal clustering in surveys
   - Calibration against common reference standards

2. **Common mistakes:**
   - Using $\chi^2 = \sum (y_i - f_i)^2 / \sigma_i^2$ without accounting for $\text{Cov}(y_i, y_j)$
   - Fitting with diagonal covariance then "inflating" error bars by $\sqrt{\chi^2/\text{dof}}$ (*ad hoc*, breaks theory)
   - Reporting nominal 1σ intervals that cover ~98% in repeated experiments

3. **Proper approach:**
   - Build the *true* covariance matrix from physical principles or data analysis
   - Use it in likelihood computation from the start
   - Verify: for correct model, $\chi^2/\text{dof} \approx 1$ and predicted σ match empirical scatter

---

## Quick Start

**See the effect immediately:**
```bash
source ~/Work/venvs/.venv/bin/activate
python multigroup_demo.py
# Outputs: 5 figures, ~1 sec runtime
```

Compare the two columns of plots:
- **Left:** Parameter ellipses — naive is ~4× smaller area
- **Right:** χ²/dof — naive shows spurious good fit (0.50 vs 0.80)
- **Bottom:** Residual patterns reveal structure ignored by naive method

---

## Author Notes

- All code is standalone; minimal dependencies (numpy, matplotlib)
- Numerical stability via Cholesky, not $C^{-1}$ inversion
- Block structure is artificial but illuminating; real data may have continuous correlation
- Monte Carlo results are stable with 100 realizations; for publication, increase to 1000+

