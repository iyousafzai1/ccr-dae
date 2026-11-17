"""Unrestricted (unmasked) decode diagnostics for the full CCR-DAE model.

Answers: how often does the UNRESTRICTED top-1 / top-10 of CCR-DAE contain
configurations for which the held-out dataset has no measured outcome?

This does not alter any reported metric. It only inspects the score vector
before the validity mask is applied, and records:
  * whether the unrestricted argmax is a measured configuration,
  * how many of the unrestricted top-10 are unmeasured,
  * how far down the unrestricted ranking the first measured configuration sits.

The masked decode is, by construction, identical to
"rank all configurations, propose them in order, skip any that yield no result".
`masked_CA` below is recomputed as a consistency check against
results/v2_per_dataset.csv.

Produces:
  results/unmasked_decode_per_dataset.csv
  results/unmasked_decode_summary.csv
"""
import _threads  # noqa: F401  (pins OMP_NUM_THREADS=32 before numpy/lightgbm load)
import numpy as np
import pandas as pd

import config as C
from data_io import Dataset
from recommenders_v2 import run_fold
from evaluate import decode_and_score

K = 10
METHOD = "ccr_dae_v2"


def main():
    data = Dataset()
    n_cfg = len(data.configs)
    rows = []

    for fold in sorted(data.test_by_fold):
        for method, d, scores in run_fold(data, fold):
            if method != METHOD:
                continue
            i = data.di[d]
            obs = data.obs[i]
            order = np.argsort(-scores)          # unrestricted ranking over all configs
            topk = order[:K]

            rows.append(dict(
                dataset=d,
                fold=fold,
                n_configs=n_cfg,
                n_measured=int(obs.sum()),
                n_unmeasured=int((~obs).sum()),
                top1_measured=bool(obs[order[0]]),
                n_unmeasured_in_top10=int((~obs[topk]).sum()),
                first_measured_rank=int(np.argmax(obs[order])) + 1,
                masked_CA=decode_and_score(scores, d, data)["CA"],
            ))

    df = pd.DataFrame(rows)
    C.RESULTS.mkdir(exist_ok=True)
    df.to_csv(C.RESULTS / "unmasked_decode_per_dataset.csv", index=False)

    n = len(df)
    summary = dict(
        n_datasets=n,
        matrix_density_pct=100.0 * df.n_measured.sum() / df.n_configs.sum(),
        top1_unmeasured_count=int((~df.top1_measured).sum()),
        top1_unmeasured_pct=100.0 * (~df.top1_measured).mean(),
        datasets_with_any_unmeasured_in_top10=int((df.n_unmeasured_in_top10 > 0).sum()),
        mean_unmeasured_in_top10=df.n_unmeasured_in_top10.mean(),
        median_unmeasured_in_top10=float(df.n_unmeasured_in_top10.median()),
        max_unmeasured_in_top10=int(df.n_unmeasured_in_top10.max()),
        mean_first_measured_rank=df.first_measured_rank.mean(),
        max_first_measured_rank=int(df.first_measured_rank.max()),
        pooled_masked_ACA=100.0 * df.masked_CA.mean(),
    )
    pd.DataFrame([summary]).to_csv(C.RESULTS / "unmasked_decode_summary.csv", index=False)

    print("\n=== Unrestricted decode diagnostics (CCR-DAE, %d datasets) ===" % n)
    for k, v in summary.items():
        print(f"  {k:40s} {v}")
    print("\n  distribution of unmeasured configs within unrestricted top-10:")
    vc = df.n_unmeasured_in_top10.value_counts().sort_index()
    for k, v in vc.items():
        print(f"    {k:2d} unmeasured : {v:3d} datasets")
    print("\n  distribution of first_measured_rank:")
    vc2 = df.first_measured_rank.value_counts().sort_index()
    for k, v in vc2.items():
        print(f"    rank {k:3d} : {v:3d} datasets")


if __name__ == "__main__":
    main()
