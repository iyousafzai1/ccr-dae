"""Direct regression baselines: RF, KRR, GB-Direct, Ridge.

Checkpoint-resumable: saves after every (method, fold) pair.

Produces:
  results/baselines_regression.csv          per (method, dataset) held-out scores
  results/baselines_regression_pooled.csv   per-method pooled ACA/ARA/HR/NDCG
"""
import _threads  # noqa: F401  (pins OMP_NUM_THREADS=32 before numpy/lightgbm load)
import os, time, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import (HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline

import config as C
from data_io import Dataset
from evaluate import decode_and_score, pooled_metrics

CKPT = C.RESULTS / "baselines_regression_checkpoint.json"
BASELINE_NAMES = ['RF_Direct', 'KRR_Direct', 'GB_Direct', 'Ridge_Direct']


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


def run_fold_baselines(data, fold):
    M, obs, CD = data.M, data.obs, data.CD
    tr_i, te_i = data.train_test_idx(fold)
    tr = [data.datasets[i] for i in tr_i]
    te = [data.datasets[i] for i in te_i]

    med = data.MF.iloc[tr_i].median()
    trf = data.MF.iloc[tr_i].fillna(med).fillna(0).values
    tef = data.MF.iloc[te_i].fillna(med).fillna(0).values
    sc = StandardScaler().fit(trf)
    Xtr = np.nan_to_num(sc.transform(trf))
    Xte = np.nan_to_num(sc.transform(tef))
    pca = PCA(C.META_PCA, random_state=C.RANDOM_STATE).fit(Xtr)
    Ptr, Pte = pca.transform(Xtr), pca.transform(Xte)

    Fx_train, Fy_train = [], []
    for a, d_name in enumerate(tr):
        i = data.di[d_name]; meas = np.where(obs[i])[0]
        for c in meas:
            Fx_train.append(np.concatenate([Ptr[a], CD[c]]))
            Fy_train.append(M[i, c])
    Fx_train, Fy_train = np.array(Fx_train), np.array(Fy_train)
    n_features = Fx_train.shape[1]

    models = {
        'RF_Direct': RandomForestRegressor(n_estimators=500, max_features='sqrt',
                        min_samples_leaf=5, n_jobs=-1, random_state=C.RANDOM_STATE),
        'KRR_Direct': make_pipeline(
                        Nystroem(kernel='rbf', gamma=1.0/n_features,
                                 n_components=min(500, len(Fx_train)),
                                 random_state=C.RANDOM_STATE),
                        Ridge(alpha=1.0)),
        'GB_Direct': HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                        max_depth=6, random_state=C.RANDOM_STATE),
        'Ridge_Direct': Ridge(alpha=10.0),
    }
    for m in models.values():
        m.fit(Fx_train, Fy_train)

    n_configs = len(data.configs)
    for a, d_name in enumerate(te):
        feat_test = np.concatenate([np.tile(Pte[a], (n_configs, 1)), CD], axis=1)
        for method_name, model in models.items():
            yield method_name, d_name, model.predict(feat_test)


def main():
    data = Dataset()
    done, rows = _load_checkpoint()
    if done:
        print(f"  Resuming: {len(done)} (fold) batches already done, {len(rows)} rows cached")
    t0 = time.time()

    for fold in sorted(data.test_by_fold):
        key = ("all_baselines", str(fold))
        if key in done:
            continue
        ft = time.time()
        for method, d, scores in run_fold_baselines(data, fold):
            sc = decode_and_score(scores, d, data)
            rows.append(dict(method=method, fold=fold, dataset=d, **sc))
        done.add(key)
        _save_checkpoint(done, rows)
        print(f"    fold={fold}  ({time.time()-ft:.1f}s)  [ckpt saved]", flush=True)

    C.RESULTS.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "baselines_regression.csv", index=False)

    print(f"\n{'Method':25s} {'ACA':>8s} {'ARA':>8s} {'HR@10':>8s} {'NDCG@10':>8s}")
    print("-" * 60)
    pooled_rows = []
    for method in BASELINE_NAMES:
        mdf = df[df.method == method]
        recs = [(None, row.to_dict()) for _, row in mdf.iterrows()]
        p = pooled_metrics(recs)
        pooled_rows.append(dict(method=method, **p))
        print(f"  {method:23s} {p['ACA']:7.2f}% {p['ARA']:7.2f}% {p['HR']:7.2f}% {p['NDCG10']:7.4f}")
    pd.DataFrame(pooled_rows).to_csv(C.RESULTS / "baselines_regression_pooled.csv", index=False)
    print(f"\nTotal: {time.time()-t0:.1f}s | Wrote baselines_regression.csv + baselines_regression_pooled.csv")


if __name__ == "__main__":
    main()
