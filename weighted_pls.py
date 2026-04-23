# -*- coding: utf-8 -*-
"""
weighted_pls.py
===============
True sample-weighted PLS regression using the weighted NIPALS algorithm.

Mathematical basis
------------------
Standard NIPALS PLS finds directions that maximise the covariance between
X-scores and y through unweighted inner products.  Here every inner product
is replaced by a *weighted* version:

    <a, b>_w  =  a' diag(w) b

This means the sample-weight vector w enters:

  1. Centering      : weighted means  mu_X = (w @ X) / sum(w)
  2. Cross-cov dir  : w_k = X0' diag(w) y0  (normalised)
  3. Score SS       : t_k' diag(w) t_k
  4. X loading      : p_k = X0' diag(w) t_k  /  (t_k' diag(w) t_k)
  5. y loading      : q_k = y0' diag(w) t_k  /  (t_k' diag(w) t_k)
  6. Deflation      : X0 -= outer(t_k, p_k),  y0 -= t_k * q_k
                      (same form as unweighted; weights enter through p_k, q_k)

New-sample transform uses the rotation matrix R = W (P'W)^{-1}, which maps
centred X to latent scores without iterative deflation.

Caveats
-------
* Sign ambiguity: NIPALS components have a sign convention; this implementation
  fixes the sign so that the weighted squared-covariance is maximised (positive
  direction).  Relative to sklearn PLSRegression, signs may differ on individual
  components — the logistic calibrator absorbs this.

* Orthogonality: Latent scores are orthogonal in the *weighted* sense
  (T' diag(w) T is diagonal), not in the unweighted sense.  When all weights
  are equal this collapses to ordinary orthogonality.

* Equivalence: When all weights are equal (w_i = const), WeightedPLSRegression
  produces identical predictions to sklearn PLSRegression(**scale=False**) (up
  to sign flips).  sklearn's default scale=True divides each feature column by
  its sample std before NIPALS; WeightedPLSRegression does not, which avoids
  distorting the per-feature weighting in the weighted inner products.  For
  SERS spectra after SNV/EMSC/Normalization preprocessing this distinction is
  immaterial because preprocessing already equalises column scales.

Usage
-----
    from weighted_pls import WeightedPLSRegression

    pls = WeightedPLSRegression(n_components=3)
    pls.fit(X_train, y_train, sample_weight=w)
    T_train = pls.transform(X_train)
    T_test  = pls.transform(X_test)
    y_hat   = pls.predict(X_test)
"""

from __future__ import annotations
import numpy as np


def _weighted_mean(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted column-wise mean.  X: (n, p), w: (n,) → (p,)."""
    return (w @ X) / w.sum()


class WeightedPLSRegression:
    """
    Weighted NIPALS PLS-1 / PLS-2 regression.

    Supports single or multiple continuous targets.  For PLS-DA, pass binary
    {0, 1} labels as y; the downstream logistic calibrator converts latent
    scores to class probabilities.

    Parameters
    ----------
    n_components : int
        Number of latent components to extract (≥1).
    max_iter     : int
        Maximum NIPALS inner iterations (for multi-target PLS-2 only; PLS-1
        converges in a single pass).
    tol          : float
        Convergence tolerance for multi-target inner loop.
    """

    def __init__(self, n_components: int = 2, max_iter: int = 500, tol: float = 1e-10):
        self.n_components = n_components
        self.max_iter     = max_iter
        self.tol          = tol

        # Fitted attributes (sklearn-compatible naming where practical)
        self.x_mean_      = None   # (p,) weighted mean of X
        self.y_mean_      = None   # (q,) weighted mean of y
        self.x_weights_   = None   # (p, A) loading weights  W
        self.x_loadings_  = None   # (p, A) X-loadings       P
        self.y_loadings_  = None   # (q, A) y-loadings       Q
        self.x_scores_    = None   # (n, A) training X-scores T
        self.x_rotations_ = None   # (p, A) rotation matrix  R = W (P'W)^{-1}
        self.coef_        = None   # (p, q) regression coefficients
        self._n_features  = None
        self._n_targets   = None

    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "WeightedPLSRegression":
        """
        Fit weighted NIPALS PLS.

        Parameters
        ----------
        X             : (n_samples, n_features)
        y             : (n_samples,) or (n_samples, n_targets)
        sample_weight : (n_samples,) or None
            If None, uniform weights (equivalent to sklearn PLSRegression).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, np.newaxis]

        n, p = X.shape
        q    = y.shape[1]

        # ── Sample weights ───────────────────────────────────────────
        if sample_weight is None:
            w = np.ones(n, dtype=np.float64)
        else:
            w = np.asarray(sample_weight, dtype=np.float64).ravel()
            if len(w) != n:
                raise ValueError(
                    f"sample_weight length {len(w)} != n_samples {n}"
                )
            if np.any(w < 0):
                raise ValueError("sample_weight must be non-negative.")
        # Normalise so that sum(w) == n (preserves the scale of inner products)
        w = w * n / w.sum()

        # ── Weighted centering ───────────────────────────────────────
        x_mean = _weighted_mean(X, w)          # (p,)
        y_mean = _weighted_mean(y, w)          # (q,)
        X0 = X - x_mean                        # (n, p)  centred
        y0 = y - y_mean                        # (n, q)  centred

        # ── Allocate storage ─────────────────────────────────────────
        A = self.n_components
        Wk = np.zeros((p, A))   # loading weights
        Pk = np.zeros((p, A))   # X-loadings
        Qk = np.zeros((q, A))   # y-loadings
        Tk = np.zeros((n, A))   # X-scores

        # ── Weighted NIPALS ──────────────────────────────────────────
        for k in range(A):
            if q == 1:
                # PLS-1: cross-cov direction found in one step (SVD of a vector)
                cov = X0.T @ (w * y0.ravel())          # (p,)
                norm_cov = np.linalg.norm(cov)
                if norm_cov < 1e-14:
                    break
                w_k = cov / norm_cov                   # (p,) unit x-weight

            else:
                # PLS-2: power iteration on cross-cov matrix X0'W Y0
                u = y0[:, 0].copy()                    # initialise u = first y col
                for _ in range(self.max_iter):
                    # x-weight
                    cov = X0.T @ (w * u)               # (p,)
                    norm_cov = np.linalg.norm(cov)
                    if norm_cov < 1e-14:
                        break
                    w_k = cov / norm_cov
                    # x-score
                    t_k_tmp = X0 @ w_k                 # (n,)
                    # y-weight (left singular vector of y0)
                    cov_y = y0.T @ (w * t_k_tmp)       # (q,)
                    norm_cov_y = np.linalg.norm(cov_y)
                    if norm_cov_y < 1e-14:
                        break
                    q_k_dir = cov_y / norm_cov_y       # (q,)
                    u_new = y0 @ q_k_dir               # (n,)
                    if np.linalg.norm(u_new - u) < self.tol:
                        u = u_new
                        break
                    u = u_new

            # ── X-scores ────────────────────────────────────────────
            t_k = X0 @ w_k                             # (n,)

            # Weighted norm (sum of squares)
            t_wss = w @ (t_k ** 2)                     # scalar  t'Wt
            if t_wss < 1e-14:
                break

            # ── Loadings ────────────────────────────────────────────
            p_k = X0.T @ (w * t_k) / t_wss            # (p,)  X-loading
            q_k = y0.T @ (w * t_k) / t_wss            # (q,)  y-loading

            # ── Deflation ───────────────────────────────────────────
            X0 -= np.outer(t_k, p_k)
            y0 -= np.outer(t_k, q_k)

            # ── Store ────────────────────────────────────────────────
            Wk[:, k] = w_k
            Pk[:, k] = p_k
            Qk[:, k] = q_k
            Tk[:, k] = t_k

        # ── Rotation matrix R = W (P'W)^{-1} ────────────────────────
        # Maps centred X directly to latent scores: T = X0 @ R
        PtW = Pk.T @ Wk                                # (A, A)
        try:
            Rk = Wk @ np.linalg.inv(PtW)
        except np.linalg.LinAlgError:
            Rk = Wk @ np.linalg.pinv(PtW)

        # Regression coefficients: y_pred = X0 @ coef + y_mean
        coef = Rk @ Qk.T                               # (p, q)

        # ── Store fitted attributes ──────────────────────────────────
        self.x_mean_      = x_mean
        self.y_mean_      = y_mean
        self.x_weights_   = Wk       # (p, A) — used for feature importance
        self.x_loadings_  = Pk       # (p, A)
        self.y_loadings_  = Qk       # (q, A)
        self.x_scores_    = Tk       # (n, A)
        self.x_rotations_ = Rk       # (p, A)
        self.coef_        = coef     # (p, q)
        self._n_features  = p
        self._n_targets   = q

        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project X to latent space.

        Parameters
        ----------
        X : (n_samples, n_features)

        Returns
        -------
        T : (n_samples, n_components)
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        return (X - self.x_mean_) @ self.x_rotations_

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict y from X.

        Parameters
        ----------
        X : (n_samples, n_features)

        Returns
        -------
        y_hat : (n_samples, n_targets)
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        return (X - self.x_mean_) @ self.coef_ + self.y_mean_

    # ------------------------------------------------------------------
    def _check_is_fitted(self):
        if self.x_mean_ is None:
            raise RuntimeError("Call fit() before transform() / predict().")
