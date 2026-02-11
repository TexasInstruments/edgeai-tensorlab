"""Sparsity v3 module for model compression and optimization.

This module provides tools and utilities for implementing structured and unstructured 
sparsity in neural networks. It focuses on N:M sparsity patterns, where N out of M
weights are kept (non-zero) in each block.

The module includes:
- Sparsity filter functions for identifying compatible layers/operations
- Weight extraction functions for accessing weights to be sparsified
- Parametrization classes that implement the sparsity patterns
- High-level APIs for applying sparsity to PyTorch models
"""

from .utils import get_sparsity_nodes, register_n2m_filters, register_filter, register_n2m_filter, register_weigth_func, register_n2m_weight_funcs, register_n2m_weight_func
from .sparsity_module import SparserModule
