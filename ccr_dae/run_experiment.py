"""End-to-end experiment: grouped 5-fold, validity-masked evaluation.

Runs every method in METHODS over the grouped cold-start protocol and writes the
per-dataset and pooled metrics.

Produces:
  results/v2_per_dataset.csv     per (method, dataset) held-out scores
  results/v2_metrics_pooled.csv  per-method pooled ACA/ARA/HR/NDCG
"""
import _threads  # noqa: F401  (pins OMP_NUM_THREADS=32 before numpy/lightgbm load)
import pandas as pd
import config as C
from data_io import Dataset
from recommenders_v2 import run_fold, METHODS
from evaluate import decode_and_score, pooled_metrics


def main():
    data = Dataset()
    records = {m: [] for m in METHODS}
    for fold in sorted(data.test_by_fold):
        for method, d, scores in run_fold(data, fold):
            records[method].append((d, decode_and_score(scores, d, data)))

    # per-dataset long form
    rows = []
    for m, recs in records.items():
        for d, sc in recs:
            rows.append(dict(method=m, internal_dataset_id=d, **sc))
    C.RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(C.RESULTS / "v2_per_dataset.csv", index=False)

    # pooled metrics
    pooled = {m: pooled_metrics(recs) for m, recs in records.items()}
    pooled_df = pd.DataFrame(pooled).T
    pooled_df.index.name = "method"
    pooled_df.to_csv(C.RESULTS / "v2_metrics_pooled.csv")

    print(pooled_df.round(4).sort_values("ACA", ascending=False).to_string())
    print("\nWrote results/v2_per_dataset.csv and results/v2_metrics_pooled.csv")


if __name__ == "__main__":
    main()
