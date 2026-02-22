#!/usr/bin/env python3
"""
Gradio Interface for Audiobook Pipeline

This module is now deprecated. Please use `from audiobook_generator import create_interface`
instead. This file now serves as a thin wrapper that imports from the package.

For CLI with gradio interface, run:
    python -m audiobook_generator --gradio

For programmatic use:
    from audiobook_generator import create_interface
    demo = create_interface()
    demo.launch()
"""
import warnings
warnings.warn(
    "audiobook_gradio_ui.py is deprecated. "
    "Use 'from audiobook_generator import create_interface' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the package gradio_ui module
from audiobook_generator.gradio_ui import *  # noqa: F401,F403
