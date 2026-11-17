"""Low-rank matrix completion (SoftImpute-style iterative SVD).

Used to complete the TRAINING performance matrix only (no test datasets), giving the meta-learner a
dense, well-estimated target and a cleaner latent manifold. Completion is for LEARNING; evaluation
still scores recommendations only against measured configurations (see evaluate.py).
"""
import numpy as np


def softimpute(X, rank=15, iters=25):
    """Complete matrix X (NaN = missing) by repeated rank-r truncated-SVD reconstruction."""
    Xf = X.copy()
    colm = np.nanmean(Xf, 0)
    nan_idx = np.where(np.isnan(Xf))
    Xf[nan_idx] = np.take(colm, nan_idx[1])           # init missing with column mean
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(Xf - colm, full_matrices=False)
        s[rank:] = 0                                   # keep top-r singular values
        L = (U * s) @ Vt + colm
        Xf = X.copy()
        m = np.isnan(Xf)
        Xf[m] = L[m]                                    # refill only the missing cells
    return Xf
