"""Models package for Attn-JGNN."""

from .attn_jgnn_model import AttnJGNN, IntraClusterGAT, InterClusterGAT

__all__ = ['AttnJGNN', 'IntraClusterGAT', 'InterClusterGAT']

# Optionally import wrapper if pytorch_lightning is available
try:
    from .wrapper import AttnJGNNWrapper
    __all__.append('AttnJGNNWrapper')
except ImportError:
    pass
