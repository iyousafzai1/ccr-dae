"""Central configuration: paths and hyperparameters for CCR-DAE.

No data is shipped with this repository. Point the paths below at your own
input files (see README.md for the expected schemas)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# Meta-feature filtering — groups excluded from the pipeline
METAFEATURE_MANIFEST = DATA / "metafeature_group_manifest.csv"
EXCLUDED_GROUPS = []   # meta-feature groups (from the manifest) to drop, e.g. ["clustering"]

# Inputs
PERF_LONG   = DATA / "performance_long.csv"        # dataset, config, family, performance
METAFEAT    = DATA / "dataset_metafeatures.csv"    # dataset x meta-features
DESCRIPTORS = DATA / "config_descriptors.csv"      # config x descriptors
FOLDS       = DATA / "dataset_folds.csv"           # grouped 5-fold test assignment

# Evaluation protocol
ETA = 1.0            # hit-rate tolerance: hit iff metric >= best - ETA * std(dataset)
NDCG_K = 10

# Method hyperparameters (set to sensible defaults; NOT tuned on test folds)
COMPLETION_RANK = 15     # SoftImpute low-rank
LATENT_DIM      = 20     # DAE-warm config latent factors
META_PCA        = 40     # meta-feature PCA components
KNN_K           = 10     # meta-feature kNN neighbours
RANK_FUSE_W     = 1.3    # listwise head weight in z-scored fusion
RANDOM_STATE    = 0
