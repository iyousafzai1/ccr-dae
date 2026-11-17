# CCR-DAE — Configuration-Conditioned Residual Denoising-Autoencoder Recommender

Reference code for **"Meta-Learning with Denoised Representations for
Cold-Start Algorithm Selection"**.

CCR-DAE recommends a classification pipeline (algorithm family *and*
hyperparameters) for a dataset on which no configuration has been evaluated. It
combines four evidence sources over a historical dataset × configuration
performance matrix: a denoised latent warm start, a local neighbourhood estimate
in meta-feature space, a configuration-descriptor-conditioned residual head, and
a listwise ranking head.

This repository contains **code only**. No benchmark data, meta-features,
configuration library, fold assignments or results are distributed. To run the
pipeline you must build your own benchmark: choose the datasets, define the
configuration library, evaluate every configuration on every dataset, extract
meta-features and assign folds, then supply the files described below.

---

## Repository layout

```
ccr-dae/
├── ccr_dae/
│   ├── config.py                     paths and hyperparameters
│   ├── data_io.py                    loading of matrix, meta-features, descriptors, folds
│   ├── completion.py                 SoftImpute low-rank completion
│   ├── recommenders_v2.py            CCR-DAE and the compared recommenders
│   ├── evaluate.py                   validity-masked decoding and metrics
│   ├── run_experiment.py             grouped k-fold experiment
│   ├── run_baselines_regression.py   direct-regression baselines
│   ├── run_pmf_comparison.py         SVD vs probabilistic matrix factorisation
│   ├── run_ablation_loo.py           leave-one-out component ablation
│   ├── run_sensitivity.py            hyperparameter sensitivity sweep
│   └── run_unmasked_decode.py        unrestricted-decode diagnostics
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Required inputs (you provide these)

Place the following CSV files in `data/` (or change the paths in `config.py`).
`data_io.py` documents exactly how each is read.

| File | Expected columns |
|---|---|
| `performance_long.csv` | `internal_dataset_id, config_id, algorithm_family, mean_balanced_accuracy` — one row per measured dataset–configuration pair; unmeasured pairs are simply absent |
| `dataset_metafeatures.csv` | first column `internal_dataset_id`, then one numeric column per meta-feature |
| `metafeature_group_manifest.csv` | `feature, group` — the group of each meta-feature (used by `config.EXCLUDED_GROUPS`) |
| `config_descriptors.csv` | first column `config_id`, then descriptor columns; `complexity_level`, `expected_runtime_level`, `hardware_type`, `algorithm_family_cfg` are one-hot encoded if present |
| `dataset_folds.csv` | `fold_id, internal_dataset_id, split` — each dataset is the held-out test set in exactly one fold |

Outputs are written to `results/`, created on first run.

---

## Running

```bash
pip install -r requirements.txt
cd ccr_dae

python3 run_experiment.py              # main grouped k-fold comparison
python3 run_baselines_regression.py    # RF / KRR / GB / Ridge direct baselines
python3 run_pmf_comparison.py          # truncated SVD vs PMF for the latent stage
python3 run_ablation_loo.py            # leave-one-out component ablation
python3 run_sensitivity.py             # hyperparameter sweep
python3 run_unmasked_decode.py         # unrestricted-decode diagnostics
```

The ablation, sensitivity, PMF and baseline scripts are checkpoint-resumable.

### Evaluation protocol

Grouped cross-validation at the dataset level: each dataset is held out as an
unseen target in exactly one fold; matrix completion, preprocessing and model
fitting use training-fold datasets only, and a held-out dataset contributes only
its meta-features. Each method scores the full configuration library; the
ranking is restricted to the held-out dataset's measured configurations before
scoring (`evaluate.decode_and_score`). Metrics: ACA, ARA, HR@10 and NDCG@10.

### Hyperparameters

Set in `config.py`: `COMPLETION_RANK`, `LATENT_DIM`, `META_PCA`, `KNN_K`,
`RANK_FUSE_W`, `ETA`, `RANDOM_STATE`. Tune them for your own benchmark.

---

## Citation

```bibtex
@article{khan2026ccrdae,
  author  = {Khan, Irfan and Ayyasamy, Ramesh Kumar and Zhang, Miao
             and Alam, Saqib and Zhang, Xiaojun},
  title   = {Meta-Learning with Denoised Representations for Cold-Start
             Algorithm Selection},
  journal = {Computer Modeling in Engineering \& Sciences},
  year    = {2026}
}
```

## License

MIT; see [LICENSE](LICENSE).
