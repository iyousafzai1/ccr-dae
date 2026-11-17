"""Honest, validity-masked evaluation.

A recommendation is scored only against configurations actually MEASURED on the target dataset, so
every method covers all datasets and no recommendation is dropped as unscoreable.
"""
import numpy as np
import config as C


def ndcg_at_k(ordered_metrics, k=C.NDCG_K):
    g = np.asarray(ordered_metrics[:k], float)
    disc = 1 / np.log2(np.arange(2, len(g) + 2))
    dcg = (g * disc).sum()
    ideal = np.sort(ordered_metrics)[::-1][:k]
    idcg = (ideal * disc[:len(ideal)]).sum()
    return dcg / idcg if idcg > 0 else np.nan


def decode_and_score(scores, d, data):
    """Mask scores to measured configs, rank, and score the top pick + ranking quality."""
    i = data.di[d] if isinstance(d, str) else d
    meas = np.where(data.obs[i])[0]
    order = meas[np.argsort(-scores[meas])]          # rank measured configs by score
    mets = data.M[i, order]
    top1, b, w, sd = mets[0], data.best[i], data.worst[i], data.std[i]
    return dict(
        CA=top1,
        RA=np.clip((top1 - w) / (b - w), 0, 1) if b > w else 1.0,
        HR=1.0 if top1 >= b - C.ETA * sd - 1e-12 else 0.0,
        NDCG10=ndcg_at_k(mets, C.NDCG_K),
    )


def pooled_metrics(per_dataset_records):
    import pandas as pd
    df = pd.DataFrame([r for _, r in per_dataset_records])
    return dict(ACA=df.CA.mean() * 100, ARA=df.RA.mean() * 100,
                HR=df.HR.mean() * 100, NDCG10=df.NDCG10.mean())
