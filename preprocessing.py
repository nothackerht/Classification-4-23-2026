import numpy as np

class Preprocessing:
    """
    Fit/apply preprocessing with optional cohort-dependent EMSC reference control.

    - fit(X_train_fxN, methods, emsc_ref_mode=..., X_unlabeled_fxN=None)
        stores any needed state (e.g., emsc_reference_)
    - transform(X_any_fxN, methods)
        applies the stored state to any matrix
    """

    def __init__(self):
        self.emsc_reference_ = None

    @staticmethod
    def _as_list(methods):
        if methods is None:
            return []
        return list(methods)

    @staticmethod
    def snv(X_fxN):
        # per-spectrum SNV (safe test-time)
        mean = X_fxN.mean(axis=0, keepdims=True)
        std = X_fxN.std(axis=0, keepdims=True) + 1e-12
        return (X_fxN - mean) / std

    @staticmethod
    def normalize(X_fxN):
        # per-spectrum normalization (safe test-time)
        norms = np.linalg.norm(X_fxN, axis=0, keepdims=True) + 1e-12
        return X_fxN / norms

    @staticmethod
    def second_derivative(X_fxN):
        # simple second difference (placeholder) — replace with your exact filter if needed
        # shape preserved by padding
        d2 = np.zeros_like(X_fxN)
        d2[1:-1, :] = X_fxN[:-2, :] - 2 * X_fxN[1:-1, :] + X_fxN[2:, :]
        d2[0, :] = d2[1, :]
        d2[-1, :] = d2[-2, :]
        return d2

    @staticmethod
    def emsc(X_fxN, reference_fx1):
        """
        Minimal EMSC-like correction using a single reference spectrum.
        NOTE: If your original preprocessing.py has a specific EMSC implementation,
        you should port THAT here unchanged, but with 'reference' injectable.
        """
        ref = reference_fx1.reshape(-1, 1)
        # project each spectrum onto ref and remove scaling
        denom = (ref.T @ ref).item() + 1e-12
        coeff = (ref.T @ X_fxN) / denom  # (1, N)
        Xcorr = X_fxN - ref @ (coeff - 1.0)  # crude correction
        return Xcorr

    def fit(self, X_train_fxN, methods, emsc_ref_mode="train_mean", X_unlabeled_fxN=None):
        methods = self._as_list(methods)
        methods_norm = [m.strip() for m in methods]

        if "EMSC" in methods_norm:
            if emsc_ref_mode == "train_mean":
                self.emsc_reference_ = np.mean(X_train_fxN, axis=1)
            elif emsc_ref_mode == "train_plus_unlabeled":
                if X_unlabeled_fxN is None:
                    raise ValueError("X_unlabeled_fxN is required for emsc_ref_mode=train_plus_unlabeled")
                both = np.concatenate([X_train_fxN, X_unlabeled_fxN], axis=1)
                self.emsc_reference_ = np.mean(both, axis=1)
            else:
                raise ValueError(f"Unknown emsc_ref_mode: {emsc_ref_mode}")

        return self

    def transform(self, X_any_fxN, methods):
        methods = self._as_list(methods)
        methods_norm = [m.strip() for m in methods]

        X = X_any_fxN.copy()

        for m in methods_norm:
            if m == "SNV":
                X = self.snv(X)
            elif m == "Normalization":
                X = self.normalize(X)
            elif m == "Second Derivative":
                X = self.second_derivative(X)
            elif m == "EMSC":
                if self.emsc_reference_ is None:
                    raise RuntimeError("EMSC requested but emsc_reference_ was not fit()")
                X = self.emsc(X, reference_fx1=self.emsc_reference_)
            else:
                # ignore unknown silently or raise — your choice
                pass

        return X


def coral_align_target_to_source(X_source: np.ndarray, X_target: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    CORAL alignment:
      X_target_aligned = (X_target - mu_t) @ Ct^{-1/2} @ Cs^{1/2} + mu_s
    where Cs/Ct are covariances of source/target.

    X_source, X_target are (n, d).
    """
    Xs = np.asarray(X_source, dtype=np.float64)
    Xt = np.asarray(X_target, dtype=np.float64)

    mu_s = Xs.mean(axis=0, keepdims=True)
    mu_t = Xt.mean(axis=0, keepdims=True)

    Xs0 = Xs - mu_s
    Xt0 = Xt - mu_t

    Cs = np.cov(Xs0, rowvar=False) + eps * np.eye(Xs0.shape[1])
    Ct = np.cov(Xt0, rowvar=False) + eps * np.eye(Xt0.shape[1])

    # eigen sqrt
    es, Vs = np.linalg.eigh(Cs)
    et, Vt = np.linalg.eigh(Ct)

    Cs_sqrt = Vs @ np.diag(np.sqrt(np.maximum(es, eps))) @ Vs.T
    Ct_invsqrt = Vt @ np.diag(1.0 / np.sqrt(np.maximum(et, eps))) @ Vt.T

    Xt_aligned = Xt0 @ Ct_invsqrt @ Cs_sqrt + mu_s
    return Xt_aligned.astype(np.float32)
