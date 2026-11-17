"""Leave-one-out component ablation.

Checkpoint-resumable: saves after every (variant, fold) pair.
If interrupted, re-run and it skips already-completed work.

Produces:
  results/ablation_loo.csv          per (variant, dataset) held-out scores
  results/ablation_loo_pooled.csv   per-variant pooled ACA/ARA/HR/NDCG
"""
import _threads  # noqa: F401  (pins OMP_NUM_THREADS=32 before numpy/lightgbm load)
import os, time, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.spatial.distance import cdist
import lightgbm as lgb

import config as C
from completion import softimpute
from data_io import Dataset
from evaluate import decode_and_score, pooled_metrics

CKPT = C.RESULTS / "ablation_loo_checkpoint.json"

VARIANTS = [
    'full', 'no_completion', 'no_masking', 'no_knn',
    'no_descriptors', 'no_residual', 'no_ranking',
    'completion_only', 'masking_only',
]


def _zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else x * 0.0


def _load_checkpoint():
    if CKPT.exists():
        with open(CKPT) as f:
            ckpt = json.load(f)
        return set(tuple(k) for k in ckpt.get("done", [])), ckpt.get("rows", [])
    return set(), []


def _save_checkpoint(done, rows):
    C.RESULTS.mkdir(exist_ok=True)
    tmp = str(CKPT) + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"done": [list(k) for k in sorted(done)], "rows": rows}, f)
    os.replace(tmp, str(CKPT))


def run_fold_ablation(data, fold, variant):
    M, obs, CD = data.M, data.obs, data.CD
    tr_i, te_i = data.train_test_idx(fold)
    tr = [data.datasets[i] for i in tr_i]
    te = [data.datasets[i] for i in te_i]

    use_completion = variant not in ('no_completion', 'masking_only')
    use_knn = variant not in ('no_knn',)
    use_descriptors = variant not in ('no_descriptors',)
    use_residual = variant not in ('no_residual', 'completion_only', 'masking_only')
    use_ranking = variant not in ('no_ranking', 'completion_only', 'masking_only')
    use_stage_b = variant not in ('completion_only', 'masking_only')

    M_tr_raw = M[tr_i].copy()
    if use_completion:
        Mtr = softimpute(M_tr_raw, rank=C.COMPLETION_RANK)
    else:
        Mtr = M_tr_raw.copy()
        colm_init = np.nanmean(Mtr, axis=0)
        nan_mask = np.isnan(Mtr)
        for j in range(Mtr.shape[1]):
            Mtr[nan_mask[:, j], j] = colm_init[j]
    colm = Mtr.mean(0)

    med = data.MF.iloc[tr_i].median()
    trf = data.MF.iloc[tr_i].fillna(med).fillna(0).values
    tef = data.MF.iloc[te_i].fillna(med).fillna(0).values
    sc = StandardScaler().fit(trf)
    Xtr = np.nan_to_num(sc.transform(trf))
    Xte = np.nan_to_num(sc.transform(tef))
    pca = PCA(C.META_PCA, random_state=C.RANDOM_STATE).fit(Xtr)
    Ptr, Pte = pca.transform(Xtr), pca.transform(Xte)

    U, s, Vt = np.linalg.svd(Mtr - colm, full_matrices=False)
    V = Vt[:C.LATENT_DIM]
    ridge = Ridge(alpha=10).fit(Ptr, U[:, :C.LATENT_DIM] * s[:C.LATENT_DIM])
    sdae_tr = ridge.predict(Ptr) @ V + colm
    sdae_te = ridge.predict(Pte) @ V + colm

    if use_knn:
        Dtr = cdist(Ptr, Ptr); np.fill_diagonal(Dtr, np.inf)
        Dte = cdist(Pte, Ptr)
        def knn(Dm):
            out = np.zeros((Dm.shape[0], len(data.configs)))
            for r in range(Dm.shape[0]):
                nn = np.argsort(Dm[r])[:C.KNN_K]
                wv = 1 / (Dm[r][nn] + 1e-6)
                out[r] = (Mtr[nn] * wv[:, None]).sum(0) / wv.sum()
            return out
        sknn_tr, sknn_te = knn(Dtr), knn(Dte)

    if not use_stage_b:
        for a, d in enumerate(te):
            yield variant, d, sdae_te[a]
        return

    Fx, Fy, grp, rel = [], [], [], []
    for a, d_name in enumerate(tr):
        i = data.di[d_name]; meas = np.where(obs[i])[0]
        b, w = data.best[i], data.worst[i]
        for c in meas:
            fp = [Ptr[a]]
            if use_descriptors: fp.append(CD[c])
            fp.append([sdae_tr[a, c]])
            if use_knn: fp.append([sknn_tr[a, c]])
            Fx.append(np.concatenate(fp)); Fy.append(M[i, c])
        grp.append(len(meas))
        rel.extend(np.clip(np.floor(5 * (M[i, meas] - w) / (b - w + 1e-9)), 0, 4).astype(int).tolist())
    Fx, Fy = np.array(Fx), np.array(Fy); rel = np.array(rel)

    if use_residual:
        gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6, random_state=C.RANDOM_STATE)
        gbm.fit(Fx, Fy)
    if use_ranking:
        rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20, verbose=-1)
        rk.fit(Fx, rel, group=grp)

    for a, d_name in enumerate(te):
        n_configs = len(data.configs)
        fp = [np.tile(Pte[a], (n_configs, 1))]
        if use_descriptors: fp.append(CD)
        fp.append(sdae_te[a][:, None])
        if use_knn: fp.append(sknn_te[a][:, None])
        feat = np.concatenate(fp, axis=1)
        s_lat = sdae_te[a]
        s_res = gbm.predict(feat) if use_residual else np.zeros(n_configs)
        s_rank = rk.predict(feat) if use_ranking else np.zeros(n_configs)
        parts = [_zscore(s_lat)]
        if use_residual: parts.append(_zscore(s_res))
        if use_ranking: parts.append(C.RANK_FUSE_W * _zscore(s_rank))
        yield variant, d_name, _zscore(sum(parts))


def main():
    data = Dataset()
    done, rows = _load_checkpoint()
    if done:
        print(f"  Resuming: {len(done)} (variant,fold) pairs already done, {len(rows)} rows cached")
    t0 = time.time()
    folds = sorted(data.test_by_fold)

    for variant in VARIANTS:
        for fold in folds:
            key = (variant, str(fold))
            if key in done:
                continue
            vt = time.time()
            # The 'no_masking' variant is scored with the same benchmark-conditioned
            # decoder as every other variant: the benchmark can only score
            # configurations with a measured outcome, and the masked top-1 is, by
            # arithmetic, the first entry of the unrestricted ranking that carries a
            # measurement. Removing the mask therefore cannot change the scored
            # configuration. This says nothing about whether an unmeasured
            # configuration would run successfully at deployment.
            for _, d, scores in run_fold_ablation(data, fold, variant):
                sc = decode_and_score(scores, d, data)
                rows.append(dict(variant=variant, fold=fold, dataset=d, **sc))
            done.add(key)
            _save_checkpoint(done, rows)
            print(f"    {variant:25s} fold={fold}  ({time.time()-vt:.1f}s)  [ckpt saved]", flush=True)

    C.RESULTS.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "ablation_loo.csv", index=False)

    pooled_rows = []
    for variant in VARIANTS:
        vdf = df[df.variant == variant]
        recs = [(None, row.to_dict()) for _, row in vdf.iterrows()]
        p = pooled_metrics(recs)
        pooled_rows.append(dict(variant=variant, **p))
        print(f"  {variant:25s}  ACA={p['ACA']:.2f}%  ARA={p['ARA']:.2f}%  HR={p['HR']:.2f}%  NDCG={p['NDCG10']:.4f}")
    pd.DataFrame(pooled_rows).to_csv(C.RESULTS / "ablation_loo_pooled.csv", index=False)
    print(f"\nTotal: {time.time()-t0:.1f}s | Wrote ablation_loo.csv + ablation_loo_pooled.csv")


if __name__ == "__main__":
    main()
