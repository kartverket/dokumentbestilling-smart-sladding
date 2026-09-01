"""
Cache key -> filename for the per-document caches. See ocr_cache.py and
yolo_cache.py.

A document name reaches the caches from local filenames and from the /model
parameter `filrevisjonid`, so it is checked before it becomes a path. Real
names are the document number, with or without a .pdf suffix, and every caller
passes a bare filename. A name that is not one safe path segment is refused
instead of being trimmed into one: trimming would let two documents share a
cache file, and a cache hit for the wrong document is silent.
"""

import os
import re

# One path segment, first character not a dot: no separator, no "..", no name
# that hides itself in a listing, no NUL.
_SAFE_DOC_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")


def cache_path(cache_dir, doc_name):
    """Path to one document's cache file, inside cache_dir.

    Raises ValueError if the name cannot be a cache key.
    """
    if not isinstance(doc_name, str) or not _SAFE_DOC_NAME.fullmatch(doc_name):
        raise ValueError(f"unusable document name for the cache: {doc_name!r}")

    base = os.path.normpath(cache_dir)
    path = os.path.normpath(os.path.join(base, os.path.splitext(doc_name)[0] + ".json"))
    if not path.startswith(base + os.sep):
        raise ValueError(f"cache path outside {base}: {doc_name!r}")
    return path
