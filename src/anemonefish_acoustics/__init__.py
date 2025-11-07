"""Anemonefish Acoustics - A package for acoustic analysis of anemonefish sounds."""

from pathlib import Path
import os
from . import data
from . import models

__version__ = "0.1.0"
__author__ = "Anemonefish Acoustics Team"
__description__ = "A package for acoustic analysis of anemonefish sounds"

# Project paths - works reliably in Docker and local environments
PACKAGE_DIR = Path(__file__).parent  # src/anemonefish_acoustics
SRC_DIR = PACKAGE_DIR.parent         # src
PROJECT_ROOT = SRC_DIR.parent        # project root
MODELS_DIR = os.path.join(SRC_DIR, "model") # models folder at project root

__all__ = [
    "data",
    "models",
    "PACKAGE_DIR",
    "SRC_DIR", 
    "PROJECT_ROOT",
    "MODELS_DIR"
]
