# -*- coding: utf-8 -*-
"""
verify_weighted_pls.py
======================
Sanity checks for WeightedPLSRegression.

Test 1 - Equal weights reproduce sklearn PLSRegression predictions.
Test 2 - Unequal weights alter the fit in the expected direction.
Test 3 - Output shapes and finiteness.
Test 4 - Near-zero-weight samples have negligible influence.

Run with:
    python verify_weighted_pls.py
"""

import sys
import numpy as np
from sklearn.cross_decomposition import PLSRegression

try:
    from weighted_pls import WeightedPLSRegression
except ImportError:
    sys.exit("ERROR: weighted_pls.py not found on PYTHONPATH or in current directory.")

RNG = np.random.default_rng(0)
PASS = []
FAIL = []


def _check(name, ok, detail=""):
    if ok:
        PASS.append(name)
        print("  PASS  " + name)
    else:
        FAIL.append(name)
        msg = "  FAIL  " + name
        if detail:
            msg += "  [" + str(detail) + "]"
        print(msg)


def _sign_align(A, B):
    """Flip columns of A so each column has positive correlation with B."""
    signs = np.sign(np.einsum("ij,ij->j", A, B))
    signs[signs == 0] = 1.0
    return A * signs


# ---------------------------------------------------------------------------
# Test 1 - Equal weights reproduce sklearn PLSRegression
# ---------------------------------------------------------------------------
print("\n[Test 1] Equal weights reproduce PLSRegression(scale=False) (up to sign flips)")
# WeightedPLSRegression centres but does NOT divide by per-feature std.
# This matches PLSRegression(scale=False).  sklearn's default scale=True
# differs by design — see weighted_pls.py docstring for rationale.
n, p = 80, 40
X = RNG.standard_normal((n, p))
y = X[:, :3].sum(axis=1) + 0.3 * RNG.standard_normal(n)

A_score = 2
sk_s = PLSRegression(n_components=A_score, scale=False)
sk_s.fit(X, y)
wp_s = WeightedPLSRegression(n_components=A_score)
wp_s.fit(X, y, sample_weight=None)

T_sk = sk_s.transform(X)
T_wp = _sign_align(wp_s.transform(X), T_sk)
score_corr = np.array([np.corrcoef(T_sk[:, k], T_wp[:, k])[0, 1] for k in range(A_score)])
print("       score corr (A=2): " + str(score_corr.round(8)))
_check("1a: A=2 latent score correlations == 1.0 (tol 1e-10)",
       bool(np.all(score_corr >= 1.0 - 1e-10)),
       "corr=" + str(score_corr.round(8)))

A_pred = 4
sk_p = PLSRegression(n_components=A_pred, scale=False)
sk_p.fit(X, y)
y_sk = sk_p.predict(X).ravel()

wp_p = WeightedPLSRegression(n_components=A_pred)
wp_p.fit(X, y, sample_weight=None)
pred_corr = float(np.corrcoef(y_sk, wp_p.predict(X).ravel())[0, 1])
_check("1b: A=4 prediction correlation == 1.0 (tol 1e-10)",
       pred_corr >= 1.0 - 1e-10,
       "corr=" + str(round(pred_corr, 12)))

w_uniform = np.full(n, 2.5)
wp_pu = WeightedPLSRegression(n_components=A_pred)
wp_pu.fit(X, y, sample_weight=w_uniform)
pred_corr2 = float(np.corrcoef(y_sk, wp_pu.predict(X).ravel())[0, 1])
_check("1c: uniform non-unit weights match (corr == 1.0, tol 1e-10)",
       pred_corr2 >= 1.0 - 1e-10,
       "corr=" + str(round(pred_corr2, 12)))


# ---------------------------------------------------------------------------
# Test 2 - Unequal weights: inverse-frequency weighting equalises patients
# ---------------------------------------------------------------------------
print("\n[Test 2] Unequal weights shift fit toward up-weighted samples")

# 10 patients in group A (10 spectra each) vs 10 patients in group B (1 spectrum each).
# Unweighted PLS is dominated by group A spectra count.
# 1/k_i weighted PLS gives each patient equal contribution.
n_pat_a, n_pat_b = 10, 10
reps_a,  reps_b  = 10,  1
Xa = RNG.standard_normal((n_pat_a * reps_a, p))
Xb = RNG.standard_normal((n_pat_b * reps_b, p)) + 3.0   # distinct group B
ya = np.zeros(n_pat_a * reps_a)
yb = np.ones(n_pat_b  * reps_b)
X2 = np.vstack([Xa, Xb])
y2 = np.concatenate([ya, yb])

# One weight per spectrum: patients in group A each contribute reps_a spectra,
# patients in group B each contribute reps_b spectra.
w_a = np.full(n_pat_a * reps_a, 1.0 / reps_a)   # shape (100,)
w_b = np.full(n_pat_b * reps_b, 1.0 / reps_b)   # shape (10,)
w_unequal = np.concatenate([w_a, w_b])
w_unequal = w_unequal * len(w_unequal) / w_unequal.sum()   # normalise

wp_uw = WeightedPLSRegression(n_components=1)
wp_uw.fit(X2, y2, sample_weight=w_unequal)

wp_eq = WeightedPLSRegression(n_components=1)
wp_eq.fit(X2, y2, sample_weight=None)

T_uw = wp_uw.transform(X2).ravel()
T_eq = wp_eq.transform(X2).ravel()

n_a = n_pat_a * reps_a
sep_uw = float(abs(T_uw[n_a:].mean() - T_uw[:n_a].mean()))
sep_eq = float(abs(T_eq[n_a:].mean() - T_eq[:n_a].mean()))

_check("2a: weighted fit separates groups (sep > 0)",
       sep_uw > 0,
       "sep_uw=" + str(round(sep_uw, 4)))
_check("2b: group separation is finite",
       np.isfinite(sep_uw) and np.isfinite(sep_eq),
       "sep_uw=" + str(round(sep_uw, 4)) + " sep_eq=" + str(round(sep_eq, 4)))
print("       group sep  weighted=" + str(round(sep_uw, 4)) +
      "  unweighted=" + str(round(sep_eq, 4)))


# ---------------------------------------------------------------------------
# Test 3 - Output shapes and finiteness
# ---------------------------------------------------------------------------
print("\n[Test 3] Output shapes and finiteness")
n_tr, n_te, p3, A3 = 60, 15, 30, 3
Xtr3 = RNG.standard_normal((n_tr, p3))
Xte3 = RNG.standard_normal((n_te, p3))
y3   = (Xtr3[:, 0] > 0).astype(float)
w3   = RNG.uniform(0.5, 1.5, size=n_tr)

wp3 = WeightedPLSRegression(n_components=A3)
wp3.fit(Xtr3, y3, sample_weight=w3)

T_tr3 = wp3.transform(Xtr3)
T_te3 = wp3.transform(Xte3)
pred3 = wp3.predict(Xte3)
imp3  = wp3.x_weights_

_check("3a: transform(train) shape == (n_tr, A)",
       T_tr3.shape == (n_tr, A3), str(T_tr3.shape))
_check("3b: transform(test)  shape == (n_te, A)",
       T_te3.shape == (n_te, A3), str(T_te3.shape))
_check("3c: predict shape == (n_te, 1)",
       pred3.shape == (n_te, 1), str(pred3.shape))
_check("3d: x_weights_ shape == (p, A)",
       imp3.shape == (p3, A3), str(imp3.shape))
_check("3e: all transforms finite",
       bool(np.all(np.isfinite(T_tr3))) and bool(np.all(np.isfinite(T_te3))), "")
_check("3f: predictions finite",
       bool(np.all(np.isfinite(pred3))), "")
_check("3g: x_weights_ finite",
       bool(np.all(np.isfinite(imp3))), "")

# Importance vector used downstream: x_weights_ @ coef_lat
from sklearn.linear_model import LogisticRegression
cal = LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced")
cal.fit(T_tr3, y3.astype(int), sample_weight=w3)
coef_lat = cal.coef_.ravel()
importance = wp3.x_weights_ @ coef_lat
_check("3h: importance vector (x_weights_ @ coef_lat) shape == (p,)",
       importance.shape == (p3,), str(importance.shape))
_check("3i: importance vector finite",
       bool(np.all(np.isfinite(importance))), "")


# ---------------------------------------------------------------------------
# Test 4 - Near-zero-weight samples have negligible influence
# ---------------------------------------------------------------------------
print("\n[Test 4] Near-zero-weight samples have negligible influence")
n4 = 60
X4 = RNG.standard_normal((n4, p))
y4 = (X4[:, 0] > 0).astype(float)

wp_ref = WeightedPLSRegression(n_components=2)
wp_ref.fit(X4[:40], y4[:40])
T_ref = wp_ref.transform(X4[:40])

# All 60 samples but last 20 carry negligible weight.
w4 = np.ones(n4)
w4[40:] = 1e-6
wp_zw = WeightedPLSRegression(n_components=2)
wp_zw.fit(X4, y4, sample_weight=w4)
T_zw = _sign_align(wp_zw.transform(X4[:40]), T_ref)

corr_zw = np.array([
    np.corrcoef(T_ref[:, k], T_zw[:, k])[0, 1]
    for k in range(2)
])
_check("4a: near-zero-weight samples have negligible influence (score corr >= 0.99)",
       bool(np.all(corr_zw >= 0.99)),
       "min corr=" + str(round(float(corr_zw.min()), 4)))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("Results: " + str(len(PASS)) + " passed, " + str(len(FAIL)) + " failed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
    sys.exit(1)
else:
    print("All checks passed.")
