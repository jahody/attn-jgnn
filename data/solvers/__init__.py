"""
Solver interfaces for #SAT (model counting).

Provides interfaces to exact #SAT solvers for generating ground truth labels.
"""

from .dsharp_solver import (
    compute_model_count,
    compute_model_count_from_string,
    batch_compute_model_counts,
    is_dsharp_available,
    get_solver_info,
)

__all__ = [
    'compute_model_count',
    'compute_model_count_from_string',
    'batch_compute_model_counts',
    'is_dsharp_available',
    'get_solver_info',
]
