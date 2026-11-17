"""Fødselsnummer recognition in OCR text: app/paddle_ocr_model_fnr.py."""
import pytest

from paddle_ocr_model_fnr import er_fnr_form, find_fnr, valid_mod11
from synthetic import break_check_digit, make_fnr, make_orgnr


def found(text, require_mod11=True):
    return [text[h.start:h.end] for h in find_fnr(text, require_mod11)]


def test_generated_numbers_pass_mod11():
    for day, month in ((1, 1), (15, 6), (31, 12)):
        assert valid_mod11(make_fnr(day, month))
    assert valid_mod11(make_fnr(d_nummer=True))


def test_every_single_digit_change_breaks_mod11():
    number = make_fnr()
    for i, digit in enumerate(number):
        for other in "0123456789":
            if other != digit:
                assert not valid_mod11(number[:i] + other + number[i + 1:])


@pytest.mark.parametrize("value", ["", "1234567890", "123456789012", "0101550000x"])
def test_mod11_needs_exactly_eleven_digits(value):
    assert not valid_mod11(value)


def test_fnr_form_accepts_fnr_and_d_nummer():
    assert er_fnr_form(make_fnr(31, 12))
    assert er_fnr_form(make_fnr(1, 1, d_nummer=True))


@pytest.mark.parametrize("head", ["000155", "320155", "720155", "011355", "010055"])
def test_fnr_form_rejects_impossible_dates(head):
    assert not er_fnr_form(head + "50000")


def test_find_plain_number():
    fnr = make_fnr()
    assert found(fnr) == [fnr]


@pytest.mark.parametrize("template", [
    "{d}{m}{y} {n}",
    "{d}.{m}.{y} {n}",
    "{d}{m}{y}-{n}",
    "{d}. {m}. {y} {n}",
])
def test_find_written_forms(template):
    fnr = make_fnr()
    text = template.format(d=fnr[:2], m=fnr[2:4], y=fnr[4:6], n=fnr[6:])
    assert found(text) == [text]


def test_find_inside_text():
    fnr = make_fnr()
    text = f"Eier: Kari Nordmann, f.nr. {fnr}, adr. Gate 4"
    hits = find_fnr(text)
    assert len(hits) == 1
    assert text[hits[0].start:hits[0].end] == fnr


def test_find_with_ocr_letter_misreads():
    fnr = make_fnr()        # day 01, month 01, year 55: has 0, 1 and 5
    misread = fnr.replace("0", "O", 1).replace("1", "l", 1).replace("5", "S", 1)
    assert misread != fnr
    assert found(misread) == [misread]


def test_find_two_numbers_on_one_line():
    first, second = make_fnr(), make_fnr(skip=1)
    assert first != second
    assert found(f"{first} og {second}") == [first, second]


def test_wrong_check_digit_is_only_a_candidate():
    bad = break_check_digit(make_fnr())
    assert found(bad) == []
    assert found(bad, require_mod11=False) == [bad]


@pytest.mark.parametrize("template", [
    "{d}{m}{y} x {n}",          # a letter in a gap
    "{d}{m}{y}   {n}",          # a gap of three characters
    "{d} {m} {y} {n1} {n2}",    # four gaps
    "{d}{m}{y}/{n}",            # a separator that is not allowed
])
def test_reject_bad_gaps(template):
    fnr = make_fnr()
    text = template.format(d=fnr[:2], m=fnr[2:4], y=fnr[4:6], n=fnr[6:],
                           n1=fnr[6:8], n2=fnr[8:])
    assert found(text) == []
    assert found(text, require_mod11=False) == []


def test_orgnr_is_not_an_fnr():
    orgnr = make_orgnr()
    assert found(f"Org.nr. {orgnr}", require_mod11=False) == []
