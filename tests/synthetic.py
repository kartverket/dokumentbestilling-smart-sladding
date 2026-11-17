"""Synthetic test data: fødselsnummer, orgnr and OCR tokens.

The numbers are computed here and never written out, since utils/fnr_vakt.py
stops a valid fødselsnummer in a tracked file. Birth years are 1855-1860
(individnummer 500-749), years before the register existed, so no real person
has one of them. The check digits are computed independently of app/, so the
tests do not grade the code against itself.
"""
from paddle_ocr_model_fnr import Token

_W1 = (3, 7, 6, 1, 8, 9, 4, 5, 2)
_W2 = (5, 4, 3, 2, 7, 6, 5, 4, 3, 2)

# 12 pt text at 300 DPI.
CHAR_W = 28    # px
LINE_H = 50    # px


def _check(weights, digits):
    return (11 - sum(w * d for w, d in zip(weights, digits)) % 11) % 11


def make_fnr(day=1, month=1, year=55, d_nummer=False, skip=0):
    """An 11-digit number with a valid date and valid check digits.

    skip picks a later individnummer, for a second distinct number.
    """
    for individ in range(500, 750):
        head = f"{day + (40 if d_nummer else 0):02d}{month:02d}{year:02d}{individ:03d}"
        digits = [int(c) for c in head]
        k1 = _check(_W1, digits)
        if k1 == 10:
            continue
        k2 = _check(_W2, digits + [k1])
        if k2 == 10:
            continue
        if skip:
            skip -= 1
            continue
        return f"{head}{k1}{k2}"
    raise ValueError("no valid number for that date")


def break_check_digit(number):
    """The same number with a wrong last check digit."""
    return number[:-1] + str((int(number[-1]) + 1) % 10)


def make_orgnr():
    """A 9-digit orgnr starting with 9 with a valid check digit."""
    weights = (3, 2, 7, 6, 5, 4, 3, 2)
    for n in range(91000000, 92000000):
        head = str(n)
        rest = sum(w * int(c) for w, c in zip(weights, head)) % 11
        check = 0 if rest == 0 else 11 - rest
        if check < 10:
            return head + str(check)


def line(*words, x=200, y=1000, rec=0.999, height=LINE_H):
    """OCR tokens for one text line: one token per word, one space apart."""
    tokens = []
    for word in words:
        x1 = x + CHAR_W * len(word)
        tokens.append(Token(word, x, y, x1, y + height, rec))
        x = x1 + CHAR_W
    return tokens


def span(token, first, last):
    """Pixel box over characters first..last (inclusive) of a token."""
    return (token.x0 + CHAR_W * first, token.y0,
            token.x0 + CHAR_W * (last + 1), token.y1)
