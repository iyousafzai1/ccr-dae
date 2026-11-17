"""Truncated SVD vs probabilistic matrix factorisation for the latent stage.

Checkpoint-resumable: saves after every (factorization, fold) pair.

Produces:
  results/pmf_comparison.csv          per (method, dataset) held-out scores
  results/pmf_comparison_pooled.csv   per-method pooled ACA/ARA/HR/NDCG
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

CKPT = C.RESULTS / "pmf_comparison_checkpoint.json"


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


def _zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else x * 0.0


def pmf_factorize(M_completed, rank, n_iter=200, lr=0.005, reg=0.02, seed=0):
    rng = np.random.RandomState(seed)
    n, m = M_completed.shape
    colm = M_completed.mean(0)
    R = M_completed - colm
    U = rng.normal(0, 0.1, (n, rank))
    V = rng.normal(0, 0.1, (rank, m))
    for it in range(n_iter):
        pred = U @ V; err = R - pred
        U -= lr * (-err @ V.T + reg * U)
        V -= lr * (-U.T @ err + reg * V)
        if it % 50 == 0 and np.sqrt((err**2).mean()) < 1e-6:
            break
    return U, V, colm


def run_fold_pmf(data, fold, factorization='svd'):
    M, obs, CD = data.M, data.obs, data.CD
    tr_i, te_i = data.train_test_idx(fold)
    tr = [data.datasets[i] for i in tr_i]
    te = [data.datasets[i] for i in te_i]

    Mtr = softimpute(M[tr_i], rank=C.COMPLETION_RANK)
    colm = Mtr.mean(0)

    med = data.MF.iloc[tr_i].median()
    trf = data.MF.iloc[tr_i].fillna(med).fillna(0).values
    tef = data.MF.iloc[te_i].fillna(med).fillna(0).values
    sc = StandardScaler().fit(trf)
    Xtr = np.nan_to_num(sc.transform(trf))
    Xte = np.nan_to_num(sc.transform(tef))
    pca = PCA(C.META_PCA, random_state=C.RANDOM_STATE).fit(Xtr)
    Ptr, Pte = pca.transform(Xtr), pca.transform(Xte)

    h = C.LATENT_DIM
    if factorization == 'svd':
        U_full, s_full, Vt_full = np.linalg.svd(Mtr - colm, full_matrices=False)
        V = Vt_full[:h]; latent_codes = U_full[:, :h] * s_full[:h]
    elif factorization == 'pmf':
        U_pmf, V_pmf, _ = pmf_factorize(Mtr, rank=h, n_iter=300, lr=0.003,
                                          reg=0.01, seed=C.RANDOM_STATE + fold)
        V = V_pmf[:h]; latent_codes = U_pmf[:, :h]
    else:
        raise ValueError(factorization)

    ridge = Ridge(alpha=10).fit(Ptr, latent_codes)
    sdae_tr = ridge.predict(Ptr) @ V + colm
    sdae_te = ridge.predict(Pte) @ V + colm

    Dtr = cdist(Ptr, Ptr); np.fill_diagonal(Dtr, np.inf); Dte = cdist(Pte, Ptr)
    def knn(Dm):
        out = np.zeros((Dm.shape[0], len(data.configs)))
        for r in range(Dm.shape[0]):
            nn = np.argsort(Dm[r])[:C.KNN_K]; wv = 1/(Dm[r][nn]+1e-6)
            out[r] = (Mtr[nn]*wv[:,None]).sum(0)/wv.sum()
        return out
    sknn_tr, sknn_te = knn(Dtr), knn(Dte)

    Fx, Fy, grp, rel = [], [], [], []
    for a, d_name in enumerate(tr):
        i = data.di[d_name]; meas = np.where(obs[i])[0]; b, w = data.best[i], data.worst[i]
        for c in meas:
            Fx.append(np.concatenate([Ptr[a], CD[c], [sdae_tr[a,c], sknn_tr[a,c]]])); Fy.append(M[i,c])
        grp.append(len(meas))
        rel.extend(np.clip(np.floor(5*(M[i,meas]-w)/(b-w+1e-9)),0,4).astype(int).tolist())
    Fx, Fy = np.array(Fx), np.array(Fy); rel = np.array(rel)

    gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6, random_state=C.RANDOM_STATE)
    gbm.fit(Fx, Fy)
    rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20, verbose=-1)
    rk.fit(Fx, rel, group=grp)

    method_name = f"CCR-DAE_{factorization.upper()}"
    for a, d_name in enumerate(te):
        feat = np.concatenate([np.tile(Pte[a],(len(data.configs),1)), CD, sdae_te[a][:,None], sknn_te[a][:,None]], 1)
        ccr = _zscore(_zscore(sdae_te[a]) + _zscore(gbm.predict(feat)) + C.RANK_FUSE_W*_zscore(rk.predict(feat)))
        yield method_name, d_name, ccr


def main():
    data = Dataset()
    done, rows = _load_checkpoint()
    if done:
        print(f"  Resuming: {len(done)} pairs done, {len(rows)} rows cached")
    t0 = time.time()

    for fac in ['svd', 'pmf']:
        for fold in sorted(data.test_by_fold):
            key = (fac, str(fold))
            if key in done:
                continue
            ft = time.time()
            for method, d, scores in run_fold_pmf(data, fold, fac):
                sc = decode_and_score(scores, d, data)
                rows.append(dict(method=method, fold=fold, dataset=d, **sc))
            done.add(key)
            _save_checkpoint(done, rows)
            print(f"    {fac.upper():5s} fold={fold}  ({time.time()-ft:.1f}s)  [ckpt saved]", flush=True)

    C.RESULTS.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "pmf_comparison.csv", index=False)

    print(f"\n{'Method':25s} {'ACA':>8s} {'ARA':>8s} {'HR@10':>8s} {'NDCG@10':>8s}")
    print("-" * 60)
    pooled_rows = []
    for method in df.method.unique():
        mdf = df[df.method == method]
        recs = [(None, row.to_dict()) for _, row in mdf.iterrows()]
        p = pooled_metrics(recs)
        pooled_rows.append(dict(method=method, **p))
        print(f"  {method:23s} {p['ACA']:7.2f}% {p['ARA']:7.2f}% {p['HR']:7.2f}% {p['NDCG10']:7.4f}")
    pd.DataFrame(pooled_rows).to_csv(C.RESULTS / "pmf_comparison_pooled.csv", index=False)
    print(f"\nTotal: {time.time()-t0:.1f}s | Wrote pmf_comparison.csv + pmf_comparison_pooled.csv")


if __name__ == "__main__":
    main()
