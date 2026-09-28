"""
Rectangle math shared by the model and the analysis tools.

A box is (x0, y0, x1, y1) with x0 <= x1 and y0 <= y1. Nothing here knows which
coordinate system the numbers are in, so both boxes in a call must be in the
same one. Mixing pixels with points is a caller bug that surfaces as a
plausible-looking share instead of an error.

The shares differ only in what they divide by, and that divisor is what a
threshold on top of them means:

    covered_share(a, b)   how much of a lies inside b. Asymmetric, so the
                          order matters: "did the prediction cover the truth
                          box" is covered_share(truth, prediction).
    smallest_share(a, b)  overlap against the smaller box. Symmetric, and
                          reaches 1.0 when one box contains the other, which
                          is what deduplication needs.

Callers that need the raw intersection as well as a share, such as the two
matchers in utils/, call intersection_area once and divide themselves rather
than computing the overlap twice.

Kjør:
    from geometry import intersection_area, area, covered_share, smallest_share
"""


def intersection_area(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    return (ix1 - ix0) * (iy1 - iy0) if (ix1 > ix0 and iy1 > iy0) else 0.0


def area(a):
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def covered_share(a, b):
    """Share of a's area that lies inside b. 0.0 when a has no area."""
    aa = area(a)
    return intersection_area(a, b) / aa if aa else 0.0


def smallest_share(a, b):
    """Overlap as a share of the smaller box. 0.0 when either box is empty."""
    least = min(area(a), area(b))
    return intersection_area(a, b) / least if least else 0.0
