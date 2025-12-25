"""
Fake News Detection ML - Source Package

This package contains the core modules for fake news detection:
- config: Configuration and hyperparameters
- data_io: Data loading and processing
- features: Feature extraction (TF-IDF, metadata)
- models_baseline: ML model implementations
- preprocess: Text preprocessing
- train: Training pipeline
- evaluate: Model evaluation
- utils: Utility functions
"""

__version__ = "1.0.0"
__author__ = "Tajaddin Gafarov"

# Lazy imports - modules are imported when accessed
__all__ = [
    "config",
    "data_io",
    "features",
    "models_baseline",
    "preprocess",
    "train",
    "evaluate",
    "utils",
]
