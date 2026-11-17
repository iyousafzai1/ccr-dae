"""Pin the OpenMP thread count before any numerical library is imported.

The gradient-boosting heads build histograms through thread-parallel
reductions whose summation order depends on the thread count, so fixing it
makes repeated runs on the same machine deterministic. Override with the
OMP_NUM_THREADS environment variable.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "32")
