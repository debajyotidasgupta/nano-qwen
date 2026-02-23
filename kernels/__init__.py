"""Triton GPU kernels for performance-critical operations.

Each kernel module provides a Triton-accelerated implementation alongside a
pure-PyTorch fallback.  The public API of every module auto-selects the Triton
path when the ``triton`` package is importable and falls back otherwise.
"""
