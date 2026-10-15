"""Datasets, leakage-safe splits and the episodic sampler.

Nothing is re-exported here on purpose: every call site imports from the specific
module (``arc.data.base``, ``arc.data.sources``, ``arc.data.medical``), which keeps the
heavy torchvision import out of the path for code that only needs ``Episode``.
"""
