"""Data loading: performance matrix, meta-features, config descriptors, folds."""
import numpy as np, pandas as pd
import config as C


class Dataset:
    def __init__(self):
        perf = pd.read_csv(C.PERF_LONG)
        self.datasets = sorted(perf.internal_dataset_id.unique())
        self.configs = sorted(perf.config_id.unique())
        self.di = {d: i for i, d in enumerate(self.datasets)}
        self.ci = {c: i for i, c in enumerate(self.configs)}

        # performance matrix M[dataset, config] = mean balanced accuracy (NaN = not measured)
        M = np.full((len(self.datasets), len(self.configs)), np.nan)
        for r in perf.itertuples():
            M[self.di[r.internal_dataset_id], self.ci[r.config_id]] = r.mean_balanced_accuracy
        self.M = M
        self.obs = ~np.isnan(M)
        self.best = np.nanmax(M, 1)
        self.worst = np.nanmin(M, 1)
        self.std = np.nanstd(M, 1)

        # dataset meta-features (clean: drop all-NaN cols, replace inf)
        mf = pd.read_csv(C.METAFEAT, index_col=0).reindex(self.datasets)
        mf = mf.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        self.MF = mf.loc[:, mf.notna().any()]

        # ── Meta-feature group filtering ────────────────────────────────────
        if C.EXCLUDED_GROUPS:
            manifest = pd.read_csv(C.METAFEATURE_MANIFEST)
            excluded_feats = set(manifest[
                manifest["group"].isin(C.EXCLUDED_GROUPS)
            ]["feature"])
            keep_cols = [c for c in self.MF.columns if c not in excluded_feats]
            self.MF = self.MF[keep_cols]

        # config descriptors -> one-hot numeric (per config, row-aligned to self.configs)
        cd = pd.read_csv(C.DESCRIPTORS, index_col=0).reindex(self.configs)
        cat = ["complexity_level", "expected_runtime_level", "hardware_type", "algorithm_family_cfg"]
        self.CD = pd.get_dummies(cd, columns=[c for c in cat if c in cd.columns]).astype(float).values

        # folds: each dataset is held-out test in exactly one fold
        folds = pd.read_csv(C.FOLDS)
        self.test_by_fold = {f: set(g.internal_dataset_id) for f, g in folds.groupby("fold_id")}

    def train_test_idx(self, fold):
        test = self.test_by_fold[fold]
        tr = [self.di[d] for d in self.datasets if d not in test]
        te = [self.di[d] for d in self.datasets if d in test]
        return tr, te
