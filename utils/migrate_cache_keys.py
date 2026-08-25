"""Renames the JSON keys in the OCR and YOLO caches on disk.

The cached files carry Norwegian keys; the readers now look for English ones.
Without this every cached document silently misses and is recomputed: days
of GPU work, and no error anyone would notice. Renaming beats bumping
CACHE_VERSION because the cached content is still valid.

Failing halfway is harmless: `version` is read before any other key, so a
file not yet reached fails the version check and is recomputed, and each file
is written to a temporary and moved into place.

Keys renamed:
    versjon -> version      ocr_modell -> ocr_model     sider -> pages
    side -> page            rotasjon -> rotation        tekst -> text
    conf_gulv -> conf_floor bokser -> boxes

Run:
    python utils/migrate_cache_keys.py --dry-run $SLADD_CACHE
    python utils/migrate_cache_keys.py --apply $SLADD_CACHE
"""

import argparse
import json
import os
import sys

KEYS = {
    "versjon": "version",
    "ocr_modell": "ocr_model",
    "sider": "pages",
    "side": "page",
    "rotasjon": "rotation",
    "tekst": "text",
    "conf_gulv": "conf_floor",
    "bokser": "boxes",
}


def convert(node):
    """Returns (new_node, renamed_count)."""
    if isinstance(node, dict):
        out, count = {}, 0
        for key, value in node.items():
            new_value, sub = convert(value)
            new_key = KEYS.get(key, key)
            if new_key != key:
                count += 1
            out[new_key] = new_value
            count += sub
        return out, count
    if isinstance(node, list):
        out, count = [], 0
        for item in node:
            new_item, sub = convert(item)
            out.append(new_item)
            count += sub
        return out, count
    return node, 0


def migrate(path, apply_changes):
    """Returns 'stale', 'current' or 'unreadable' for one cache file."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return "unreadable", 0

    converted, renamed = convert(data)
    if not renamed:
        return "current", 0
    if not apply_changes:
        return "stale", renamed

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)
    return "stale", renamed


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", metavar="DIR",
                   help="cache directories, searched recursively")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--apply", action="store_true")
    args = p.parse_args()

    tally = {"stale": 0, "current": 0, "unreadable": 0}
    renamed_total = 0
    unreadable = []

    for root in args.roots:
        for base, _dirs, files in os.walk(root):
            for name in files:
                if not name.endswith(".json"):
                    continue
                path = os.path.join(base, name)
                status, renamed = migrate(path, args.apply)
                tally[status] += 1
                renamed_total += renamed
                if status == "unreadable":
                    unreadable.append(path)

    verb = "renamed" if args.apply else "would rename"
    print(f"{sum(tally.values())} cache file(s): {tally['stale']} to migrate, "
          f"{tally['current']} already current, {tally['unreadable']} unreadable")
    print(f"{verb} {renamed_total} key(s)")
    for path in unreadable[:20]:
        print(f"  unreadable: {path}", file=sys.stderr)
    if len(unreadable) > 20:
        print(f"  ... and {len(unreadable) - 20} more", file=sys.stderr)

    if args.dry_run and tally["stale"]:
        print("\nNothing was written. Re-run with --apply to migrate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
