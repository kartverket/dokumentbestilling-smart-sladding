"""Puts app/ and job/ on the import path, the way their containers run them."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for folder in ("app", "job"):
    sys.path.insert(0, os.path.join(ROOT, folder))
