"""Path resolution for the Part I (collective-foraging cascades) figure scripts.

Resolves the repository's Code/ and figures/ directories and a data directory,
and puts Code/ on sys.path so the scripts can import the simulation engine
(mobile_cascade) and helpers. Importing this module is enough.

Precomputed sweep/replot data (.npz) is NOT shipped in this repository. Scripts
that read cached data look under DATA (default: <repo>/data). Either:
  - set the FLOCKING_DATA environment variable to a directory holding the .npz
    files, or
  - regenerate the data with the gen_*/rq2_* scripts (these run the engine).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (experiments/ sits under it)
CODE = os.path.join(ROOT, "Code")
FIG = os.path.join(ROOT, "figures")
DATA = os.environ.get("FLOCKING_DATA", os.path.join(ROOT, "data"))

if CODE not in sys.path:
    sys.path.insert(0, CODE)
