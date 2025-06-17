"""
Blood Pressure Prediction using VAE + BiLSTM

A modular framework for blood pressure prediction using Variational Autoencoders
for dimensionality reduction and Bidirectional LSTMs for temporal modeling.
"""

__version__ = "1.0.0"
__author__ = "Lucas Takanori"

# Import modules conditionally to avoid circular imports
__all__ = []

try:
    from . import utils
    __all__.append("utils")
except ImportError:
    pass

try:
    from . import training
    __all__.append("training")
except ImportError:
    pass

try:
    from . import data
    __all__.append("data")
except ImportError:
    pass

try:
    from . import models
    __all__.append("models")
except ImportError:
    pass
