# Block Bootstrap for SF Covariance: Quick Reference

## TL;DR

**Problem:** Epoch bootstrap inflates lag-0 variance by **6.5×** and destroys covariance structure.

**Solution:** Use **block bootstrap** with block size **B = 40 epochs** (or choose yours via sensitivity study).

**Result:** Lag-0 variance inflated by only **1.32×**, all metrics within target.

---

## Metric Comparison

| Metric | Epoch Bootstrap | Block (B=40) | Target | Status |
|---|---|---|---|---|
| Lag-0 variance ratio | 6.50× | 1.32× | <1.5× | ✓ PASS |
| Variance RMSE | 0.0516 | 0.0110 | <0.020 | ✓ PASS |
| Improvement factor | baseline | 4.9× | >2× | ✓ PASS |

---

## Implementation (3 Steps)

### Step 1: Choose Your Block Size

**Formula:**
$$B = N_{\text{epoch}} \times \frac{\tau_{\max}}{T_{\text{baseline}}}$$

**Example:**
- Your data: N_epoch = 100 observations, τ_max = 5 years, T = 20 years
- B = 100 × (5/20) = 25 epochs

**Conservative fallback:** B = 0.5 × N_epoch (always safe)

### Step 2: Use Block Bootstrap Function

```python
from mock_sf_covariance_v2 import bootstrap_sf_covariance_block

# In your covariance estimation code:
cov_boot, mean_A_boot, A_boot = bootstrap_sf_covariance_block(
    times=times_of_your_lightcurve,
    mags=magnitudes,
    lags=lag_array,
    lag_width=lag_bin_width,
    n_bootstrap=500,  # or your choice
    rng=np.random.default_rng(seed),
    z=redshift,
    sigma_phot=photometric_noise,
    block_size=B  # YOUR BLOCK SIZE HERE
)
```

### Step 3: Validate on Your Data

Run the **sensitivity study** on your actual cadence:

```python
from test_block_size_sensitivity import run_sensitivity_study, print_summary

# Test a range of block sizes
block_sizes = [max(5, B-10), B, B+10]  # test around your choice
results = run_sensitivity_study(block_sizes)
best_B = print_summary(results)
```

**Check these acceptance criteria:**
- [ ] Lag-0 variance ratio < 1.5×
- [ ] Variance RMSE < 0.020
- [ ] Correlation matrix visually smooth (no random spikes)
- [ ] Variance profile follows true shape

---

## How Block Bootstrap Works

### Why Epoch Bootstrap Fails

```
Original lightcurve:  t₁, t₂, t₃, ..., t₈₀ (with magnitudes m₁, m₂, ...)
Epoch bootstrap:      t₁₅, t₂, t₁₅, t₃₀, ... (indices drawn with replacement)
                      → Creates duplicate times!
                      → Spurious zero-lag pairs (Δt = 0)
                      → SF²(lag=0) inflated
                      → All correlations corrupted
```

### Why Block Bootstrap Works

```
Original lightcurve:  [t₁...t₁₀], [t₁₁...t₂₀], [t₂₁...t₃₀], ...
Block bootstrap:      [t₂₁...t₃₀], [t₆₁...t₇₀], [t₁₁...t₂₀], ...
                      → Blocks are real segments from original
                      → No spurious duplicates within blocks
                      → Local structure (lags ≤ B) preserved exactly
                      → Only short-term artifacts at block boundaries
```

---

## Files You Need

| File | Purpose | Use When |
|---|---|---|
| `mock_sf_covariance_v2.py` | Core bootstrap functions | Implementing block bootstrap in your code |
| `test_block_size_sensitivity.py` | Sensitivity study | Choosing optimal B for your cadence |
| `test_block_bootstrap.py` | Detailed validation | Comparing epoch vs. block bootstrap |
| `BOOTSTRAP_ANALYSIS.md` | Full technical documentation | Understanding theory and assumptions |

---

## Example: Find Optimal B for Your Data

```python
# test_block_size_sensitivity.py with your data

from test_block_size_sensitivity import run_sensitivity_study, print_summary

# 1. Define your observational parameters
n_epochs = 120  # from your survey
t_baseline = 10.0  # years
tau_max = 3.0  # your largest lag of interest

# 2. Estimate reasonable block sizes to test
B_nominal = int(n_epochs * (tau_max / t_baseline))  # e.g., 36
block_sizes = [max(5, B_nominal - 15),
               B_nominal,
               B_nominal + 15]  # e.g., [21, 36, 51]

# 3. Run sensitivity study
results = run_sensitivity_study(block_sizes)

# 4. Get recommendation
best_B = print_summary(results)

# 5. Use best_B in your pipeline
```

---

## Diagnostics: What to Check

### 1. Lag-0 Variance Ratio
- **Metric:** diag(Σ̂_boot)[0] / diag(Σ_true)[0]
- **Target:** < 1.5×
- **What it means:** Variance at shortest lag; most sensitive diagnostic

### 2. Variance RMSE
- **Metric:** √(mean((diag(Σ̂) - diag(Σ_true))²))
- **Target:** < 0.020
- **What it means:** Overall accuracy of variance estimates across all lags

### 3. Correlation Matrix
- **Visual check:** Should show smooth decay away from diagonal
- **Bad sign:** Random positive/negative spikes → B too small
- **Good sign:** Coherent structure matching true matrix

### 4. Variance Profile
- **Visual check:** Estimated variances follow true curve
- **Bad sign:** Epoch bootstrap has huge spike at lag-0
- **Good sign:** Block bootstrap curve tracks truth

---

## Troubleshooting

### "My block bootstrap is worse than epoch?"
→ Block size B is too small. Increase B (test larger values in sensitivity study).

### "All block sizes perform similarly?"
→ Your lag range may be very short. Use more conservative B (higher) to preserve structure.

### "Lag-0 variance still > 2×?"
→ Block bootstrap cannot perfectly eliminate this. Accept 1.3-1.5× as good.

### "Covariance structure still noisy?"
→ You may need more bootstrap replicates (increase n_bootstrap) or more independent lightcurves.

---

## References

- **Block bootstrap theory:** Künsch (1989), *Annals of Statistics* 17:1217–1241
- **Practical guide:** Lahiri (1999), "Resampling Methods for Dependent Data"
- **This implementation:** Based on circular block bootstrap (Politis & Romano 1994)

---

## Questions?

See `BOOTSTRAP_ANALYSIS.md` for:
- Full technical explanation
- Alternative methods (residual bootstrap, pooled covariance)
- Theoretical justification
- Caveats and limitations
