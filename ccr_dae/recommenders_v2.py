"""CCR-DAE v2 and the recommenders compared against it.

CCR-DAE = configuration-conditioned residual latent recommender, upgraded with:
  1. low-rank matrix completion of the (train-only) performance matrix,
  2. an SVD/ridge latent warm-start base score,
  3. a meta-feature kNN head,
  4. a descriptor-conditioned residual (gradient boosting) head,
  5. a listwise LambdaRank reranker head,
  6. fusion of the heads, and validity masking at decode time (see evaluate.py).

All learning uses TRAIN datasets only within each grouped outer fold; test datasets contribute only
their meta-features. Returns, per held-out test dataset, a score vector over all configurations for
each method (decoding/masking happens in evaluate.py).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from scipy.spatial.distance import cdist
from scipy.optimize import nnls
import lightgbm as lgb

import config as C
from completion import softimpute

METHODS = ["global_best", "dae_warm_v2", "knn_v2", "gbm_residual_v2",
           "ranker_v2", "ccr_dae_v2_regfuse", "ccr_dae_v2"]


def _corrupt_train_matrix(M, obs, tr_i, epsilon, seed):
    """Corrupt observed training values: keep fraction `epsilon`, replace rest with noise."""
    M_input = M[tr_i].copy()                    # (n_train, n_configs)
    obs_tr = obs[tr_i]                           # bool mask of observed entries

    rng = np.random.RandomState(seed)
    keep = rng.binomial(1, epsilon, size=M_input.shape).astype(bool)

    col_std = np.nanstd(M_input, axis=0, keepdims=True)
    col_std = np.where(np.isfinite(col_std) & (col_std > 1e-9), col_std, 1.0)
    rng_noise = np.random.RandomState(seed + 1)
    noise = rng_noise.normal(0, 0.5, M_input.shape) * col_std

    return np.where(obs_tr & ~keep, noise, np.where(obs_tr, M_input, np.nan))


def _zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else x * 0.0


def run_fold(data, fold, epsilon=None, eps_seed_offset=0):
    """Fit all heads on the fold's training datasets; yield (method, dataset, score_vector).

    If epsilon < 1.0 is given, corrupt the TRAINING portion of the performance matrix
    (keeping only fraction epsilon of observed values, adding Gaussian noise to the rest)
    BEFORE matrix completion — enabling epsilon-sensitivity experiments.
    """
    M, obs, CD = data.M, data.obs, data.CD
    tr_i, te_i = data.train_test_idx(fold)
    tr = [data.datasets[i] for i in tr_i]
    te = [data.datasets[i] for i in te_i]

    # 1. completion (train only) — optionally with epsilon-corrupted input
    if epsilon is not None and epsilon < 1.0:
        M_input = _corrupt_train_matrix(M, obs, tr_i, epsilon,
                                        seed=C.RANDOM_STATE + fold * 100 + eps_seed_offset)
    else:
        M_input = M[tr_i].copy()
    Mtr = softimpute(M_input, rank=C.COMPLETION_RANK)
    colm = Mtr.mean(0)

    # meta-features -> scale -> PCA (fit on train)
    med = data.MF.iloc[tr_i].median()
    trf = data.MF.iloc[tr_i].fillna(med).fillna(0).values
    tef = data.MF.iloc[te_i].fillna(med).fillna(0).values
    sc = StandardScaler().fit(trf)
    Xtr = np.nan_to_num(sc.transform(trf)); Xte = np.nan_to_num(sc.transform(tef))
    pca = PCA(C.META_PCA, random_state=C.RANDOM_STATE).fit(Xtr)
    Ptr, Pte = pca.transform(Xtr), pca.transform(Xte)

    # 2. SVD/ridge latent warm-start: config factors + meta -> dataset latent code.
    U, s, Vt = np.linalg.svd(Mtr - colm, full_matrices=False)
    V = Vt[:C.LATENT_DIM]
    ridge = Ridge(alpha=10).fit(Ptr, U[:, :C.LATENT_DIM] * s[:C.LATENT_DIM])
    sdae_tr = ridge.predict(Ptr) @ V + colm
    sdae_te = ridge.predict(Pte) @ V + colm

    # 3. meta-feature kNN head
    Dtr = cdist(Ptr, Ptr); np.fill_diagonal(Dtr, np.inf); Dte = cdist(Pte, Ptr)

    def knn(Dm):
        out = np.zeros((Dm.shape[0], len(data.configs)))
        for r in range(Dm.shape[0]):
            nn = np.argsort(Dm[r])[:C.KNN_K]; wv = 1 / (Dm[r][nn] + 1e-6)
            out[r] = (Mtr[nn] * wv[:, None]).sum(0) / wv.sum()
        return out
    sknn_tr, sknn_te = knn(Dtr), knn(Dte)

    # build train pair features for residual + ranker
    Fx, Fy, grp, rel = [], [], [], []
    for a, d in enumerate(tr):
        i = data.di[d]; meas = np.where(obs[i])[0]; b, w = data.best[i], data.worst[i]
        for c in meas:
            Fx.append(np.concatenate([Ptr[a], CD[c], [sdae_tr[a, c], sknn_tr[a, c]]])); Fy.append(M[i, c])
        grp.append(len(meas))
        rel.extend(np.clip(np.floor(5 * (M[i, meas] - w) / (b - w + 1e-9)), 0, 4).astype(int).tolist())
    Fx, Fy = np.array(Fx), np.array(Fy); rel = np.array(rel)

    # 4. descriptor-conditioned residual (gradient boosting), with OOF preds for stacking
    gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6,
                                        random_state=C.RANDOM_STATE)
    oof = cross_val_predict(gbm, Fx, Fy, cv=3); gbm.fit(Fx, Fy)

    # 5. listwise LambdaRank reranker
    rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05,
                        num_leaves=31, min_child_samples=20, verbose=-1)
    rk.fit(Fx, rel, group=grp)

    # nnls stack weights for the regression fusion
    sdf, skf = [], []
    for a, d in enumerate(tr):
        meas = np.where(obs[data.di[d]])[0]; sdf.extend(sdae_tr[a, meas]); skf.extend(sknn_tr[a, meas])
    wv, _ = nnls(np.vstack([sdf, skf, oof]).T, Fy); wv = wv / (wv.sum() or 1)

    for a, d in enumerate(te):
        feat = np.concatenate([np.tile(Pte[a], (len(data.configs), 1)), CD,
                               sdae_te[a][:, None], sknn_te[a][:, None]], 1)
        sgbm = gbm.predict(feat); srank = rk.predict(feat)
        regfuse = wv[0] * sdae_te[a] + wv[1] * sknn_te[a] + wv[2] * sgbm
        ccr = _zscore(_zscore(sdae_te[a]) + _zscore(sgbm) + C.RANK_FUSE_W * _zscore(srank))
        yield "global_best", d, colm.copy()
        yield "dae_warm_v2", d, sdae_te[a]
        yield "knn_v2", d, sknn_te[a]
        yield "gbm_residual_v2", d, sgbm
        yield "ranker_v2", d, srank
        yield "ccr_dae_v2_regfuse", d, regfuse
        yield "ccr_dae_v2", d, ccr
