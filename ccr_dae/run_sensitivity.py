"""Hyperparameter sensitivity analysis.

Checkpoint-resumable: saves after every (parameter, value) sweep point.

Produces:
  results/sensitivity_sweep.csv   (parameter, value, ACA, ARA, HR, NDCG10)
"""
import _threads  # noqa: F401  (pins OMP_NUM_THREADS=32 before numpy/lightgbm load)
import os, time, json
import numpy as np
import pandas as pd

import config as C
from data_io import Dataset
from recommenders_v2 import run_fold
from evaluate import decode_and_score, pooled_metrics

CKPT = C.RESULTS / "sensitivity_checkpoint.json"

SWEEPS = {
    'COMPLETION_RANK': [5, 10, 15, 20, 30],
    'LATENT_DIM':      [5, 10, 15, 20, 30, 40],
    'KNN_K':           [3, 5, 10, 15, 20, 30],
    'RANK_FUSE_W':     [0.5, 0.8, 1.0, 1.3, 1.5, 2.0],
    'META_PCA':        [10, 20, 30, 40, 60],
}


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


def run_full_pipeline(data, **overrides):
    originals = {}
    for k, v in overrides.items():
        originals[k] = getattr(C, k); setattr(C, k, v)
    try:
        records = []
        for fold in sorted(data.test_by_fold):
            for method, d, scores in run_fold(data, fold):
                if method == 'ccr_dae_v2':
                    records.append((d, decode_and_score(scores, d, data)))
        return pooled_metrics(records)
    finally:
        for k, v in originals.items():
            setattr(C, k, v)


def main():
    data = Dataset()
    done, rows = _load_checkpoint()
    if done:
        print(f"  Resuming: {len(done)} sweep points done, {len(rows)} rows cached")
    t0 = time.time()

    for param, values in SWEEPS.items():
        for val in values:
            key = (param, str(val))
            if key in done:
                continue
            vt = time.time()
            p = run_full_pipeline(data, **{param: val})
            rows.append(dict(parameter=param, value=val, **p))
            done.add(key)
            _save_checkpoint(done, rows)
            default_marker = " *" if val == getattr(C, param) else ""
            print(f"    {param}={val}{default_marker:2s}  ACA={p['ACA']:.2f}%  ARA={p['ARA']:.2f}%  "
                  f"HR={p['HR']:.2f}%  NDCG={p['NDCG10']:.4f}  ({time.time()-vt:.1f}s)  [ckpt]", flush=True)

    C.RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(C.RESULTS / "sensitivity_sweep.csv", index=False)
    print(f"\nTotal: {time.time()-t0:.1f}s | Wrote sensitivity_sweep.csv")


if __name__ == "__main__":
    main()
