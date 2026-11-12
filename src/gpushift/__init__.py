"""gpushift: a shared GPU job queue.

Keep this module free of imports: ``gpushift.checkpoint`` is imported by user
training scripts and must stay dependency-free.
"""

__version__ = "0.1.2"
