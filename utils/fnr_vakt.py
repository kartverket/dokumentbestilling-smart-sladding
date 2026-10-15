"""Stops commits that contain valid fødselsnummer.

Only numbers that pass both mod-11 check digits AND carry a valid date are
flagged, so coordinates, dagbok numbers and ids go free. Forms covered:

    01010112345   010101 12345   010101-12345   01.01.01 12345

Three kinds stop a commit: fnr, d-nummer (day + 40) and h-nummer (month + 40).
Synthetic Tenor numbers (month + 80) pass on purpose. They are what examples
in code should use.

Run:
    python utils/fnr_vakt.py --staged      # what the pre-commit hook does
    python utils/fnr_vakt.py --all         # every tracked file
    python utils/fnr_vakt.py FILE [FILE...]
    python utils/fnr_vakt.py --selftest

On a false hit, an account number that happens to have valid check digits or
a fnr in a fixture that belongs there, write "fnr-ok" in a comment on the same
line and the whole line is skipped.
"""

import argparse
import datetime
import re
import subprocess
import sys

# dd mm yy nnnnn, with an optional single separator between the groups.
# Lookaround on both sides so 11 digits inside a longer run do not match.
CANDIDATE = re.compile(
    r"(?<![0-9])(\d{2})[ .\-]?(\d{2})[ .\-]?(\d{2})[ .\-]?(\d{5})(?![0-9])")

MOD11_WEIGHTS_1 = (3, 7, 6, 1, 8, 9, 4, 5, 2)
MOD11_WEIGHTS_2 = (5, 4, 3, 2, 7, 6, 5, 4, 3, 2)

MARKER = "fnr-ok"

# The kinds the guard stops on. "synthetic" is recognised, but let through.
STOPS_ON = ("fnr", "d-nummer", "h-nummer")


def check_digits_ok(digits):
    """True if both mod-11 check digits of an 11-digit run are correct."""
    d = [int(c) for c in digits]
    k1 = 11 - (sum(v * x for v, x in zip(MOD11_WEIGHTS_1, d[:9])) % 11)
    if k1 == 11:
        k1 = 0
    if k1 == 10 or k1 != d[9]:
        return False
    k2 = 11 - (sum(v * x for v, x in zip(MOD11_WEIGHTS_2, d[:10])) % 11)
    if k2 == 11:
        k2 = 0
    return k2 != 10 and k2 == d[10]


def kind_and_date(digits):
    """Returns (kind, date) for an 11-digit run, or (None, None).

    The individual number decides the century, so 010190 can be both 1890 and
    1990 depending on the digits behind it. If the combination falls outside
    the assigned series it is no fnr, whatever the check digits say.
    """
    day, month, yy = int(digits[0:2]), int(digits[2:4]), int(digits[4:6])
    kind = "fnr"

    if 41 <= day <= 71:
        day -= 40
        kind = "d-nummer"

    if 81 <= month <= 92:
        month -= 80
        kind = "synthetic"
    elif 41 <= month <= 52:
        month -= 40
        kind = "h-nummer"

    ind = int(digits[6:9])
    if ind <= 499:
        year = 1900 + yy
    elif 500 <= ind <= 749 and yy >= 54:
        year = 1800 + yy
    elif 500 <= ind <= 999 and yy <= 39:
        year = 2000 + yy
    elif 900 <= ind <= 999 and yy >= 40:
        year = 1900 + yy
    else:
        return None, None

    try:
        return kind, datetime.date(year, month, day)
    except ValueError:
        return None, None


def find(line):
    """Every fnr-like hit on one line: [(column, kind), ...].

    Empty list if the line carries the exemption marker.
    """
    if MARKER in line:
        return []
    hits = []
    for m in CANDIDATE.finditer(line):
        digits = "".join(m.groups())
        if not check_digits_ok(digits):
            continue
        kind, _ = kind_and_date(digits)
        if kind:
            hits.append((m.start() + 1, kind))
    return hits


def scan_text(text, path, first_line=1):
    """[(path, line, column, kind), ...] for the kinds that must stop a commit."""
    found = []
    for i, line in enumerate(text.splitlines()):
        for col, kind in find(line):
            if kind in STOPS_ON:
                found.append((path, first_line + i, col, kind))
    return found


def _git(*args):
    return subprocess.run(
        ("git", "-c", "core.quotePath=false") + args,
        capture_output=True, text=True, errors="replace").stdout


def scan_staged():
    """Scans only the lines the commit adds.

    Scanning whole files would block work on files that already carry a hit.
    The guard exists to stop new ones, not to lock the old ones.
    """
    diff = _git("diff", "--cached", "--unified=0", "--no-color",
                "--diff-filter=ACMR")
    found, path, lineno = [], None, 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            target = line[4:]
            path = None if target == "/dev/null" else target[2:]
        elif line.startswith("@@"):
            m = re.match(r"@@ -\S+ \+(\d+)", line)
            lineno = int(m.group(1)) if m else 0
        elif line.startswith("+") and not line.startswith("+++") and path:
            for col, kind in find(line[1:]):
                if kind in STOPS_ON:
                    found.append((path, lineno, col, kind))
            lineno += 1
    return found


def scan_files(paths):
    found = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                found += scan_text(f.read(), path)
        except (UnicodeDecodeError, IsADirectoryError):
            continue  # binary files and directories are none of our business
        except OSError as e:
            print(f"cannot read {path}: {e}", file=sys.stderr)
    return found


def scan_all_tracked():
    out = _git("ls-files", "-z")
    return scan_files([s for s in out.split("\0") if s])


def report(found):
    """Writes the findings and returns the exit code."""
    if not found:
        return 0

    width = max(len(f"{p}:{l}:{c}") for p, l, c, _ in found)
    print("\nFØDSELSNUMMER FOUND, commit stopped\n", file=sys.stderr)
    for path, lineno, col, kind in found:
        where = f"{path}:{lineno}:{col}"
        print(f"  {where:<{width}}  {kind}", file=sys.stderr)

    print(f"""
The numbers themselves are not printed. Open the lines above and replace them
with synthetic Tenor test numbers (month + 80), which this guard lets through.

If the hit is wrong, an account number with accidentally valid check digits
for instance, write "{MARKER}" in a comment on the same line.

If you must commit anyway:  git commit --no-verify
""", file=sys.stderr)
    return 1


# ---------------------------------------------------------------- self-test

def _make_number(day, month, yy, ind=0):
    """Builds a number with valid check digits from a date field.

    The self-test needs numbers that pass the check, and they cannot sit as
    literals in this file. The guard would stop its own source, and the repo
    would hold valid fødselsnumre in code again. They are computed here
    instead, and are gone when the process ends.
    """
    while ind <= 999:
        d = [int(c) for c in f"{day:02d}{month:02d}{yy:02d}{ind:03d}"]
        k1 = 11 - (sum(v * x for v, x in zip(MOD11_WEIGHTS_1, d)) % 11)
        k1 = 0 if k1 == 11 else k1
        if k1 != 10:
            k2 = 11 - (sum(v * x for v, x in zip(MOD11_WEIGHTS_2, d + [k1])) % 11)
            k2 = 0 if k2 == 11 else k2
            if k2 != 10:
                return "".join(str(x) for x in d) + f"{k1}{k2}"
        ind += 1
    raise AssertionError("no valid combination found")


def selftest():
    failures = []

    def check(condition, what):
        print(f"  {'ok  ' if condition else 'FAIL'}  {what}")
        if not condition:
            failures.append(what)

    fnr = _make_number(6, 6, 95)
    dnr = _make_number(46, 6, 95)         # day + 40
    hnr = _make_number(6, 46, 95)         # month + 40
    synth = _make_number(6, 86, 95)         # month + 80, Tenor
    d, m, y, p = fnr[0:2], fnr[2:4], fnr[4:6], fnr[6:]

    print("Written forms")
    for text in (fnr, f"{d}{m}{y} {p}", f"{d}{m}{y}-{p}", f"{d}.{m}.{y} {p}"):
        check(len(scan_text(f"Hjemmelshaver {text} eier 1/2", "x")) == 1,
              f"hit on \"{text[:6]}...\"")

    print("Kinds")
    check(find(f"Kjoper {dnr}")[0][1] == "d-nummer", "d-nummer is stopped")
    check(find(f"Pasient {hnr}")[0][1] == "h-nummer", "h-nummer is stopped")
    check(find(f"Testperson {synth}")[0][1] == "synthetic",
          "Tenor number is recognised as synthetic")
    check(scan_text(f"Testperson {synth}", "x") == [],
          "Tenor number does not stop the commit")

    print("No false hits")
    invalid = fnr[:10] + str((int(fnr[10]) + 1) % 10)
    check(scan_text(invalid, "x") == [], "wrong check digit goes free")
    check(scan_text("Hjemmelshaver 060695 00000", "x") == [],
          "the example value 060695 00000 goes free")
    check(scan_text("Koordinat N 6626630.58", "x") == [], "coordinate goes free")
    check(scan_text("Dagboknr 900123 tinglyst 03.11.1998", "x") == [],
          "dagbok number and date go free")
    check(scan_text(f"9{fnr}", "x") == [] and scan_text(f"{fnr}9", "x") == [],
          "11 digits inside a longer run go free")

    print("Exemption")
    check(scan_text(f"truth = {fnr}  # {MARKER}, from the test set", "x") == [],
          f"\"{MARKER}\" on the line turns the check off")

    print("Reporting")
    found = scan_text(f"a = {fnr}", "file.py")
    check(found == [("file.py", 1, 5, "fnr")],
          "path, line, column and kind are correct")
    check(str(fnr) not in repr(found), "the number itself is not in the finding")

    print()
    if failures:
        print(f"{len(failures)} failed")
        return 1
    print("All green")
    return 0


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", metavar="FILE")
    p.add_argument("--staged", action="store_true",
                   help="scan the lines staged for commit")
    p.add_argument("--all", action="store_true",
                   help="scan every tracked file in the working tree")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()

    if a.selftest:
        return selftest()
    if a.staged:
        return report(scan_staged())
    if a.all:
        return report(scan_all_tracked())
    if a.files:
        return report(scan_files(a.files))
    p.print_usage(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
