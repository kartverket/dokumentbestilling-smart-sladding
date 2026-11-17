"""From OCR tokens and YOLO boxes to sladd boxes: app/model_main.py.

Runs _find_boxes_with_source on synthetic tokens, so the box placement, the
Paddle/YOLO merge and the OCR and profile rules are tested without a GPU.
"""
from config import (RULE_DECIMAL, RULE_DECIMAL_LOW_TIER,
                    YOLO_CONF_GEOMETRY_THRESHOLD)
from model_main import _find_boxes_with_source
from yolo_fnr import has_paddle_noise_shape
from paddle_ocr_model_fnr import build_lines, sladd_boxes_from_tokens
from synthetic import CHAR_W, line, make_fnr, span


def run(tokens, yolo_boxes=(), **kwargs):
    """[(box, kilde, yolo_conf)] for one page."""
    result = _find_boxes_with_source(tokens, build_lines(tokens),
                                     list(yolo_boxes), **kwargs)
    return [(box, kilde, conf) for box, kilde, conf, _rec, _features in result]


def kilder(tokens, yolo_boxes=(), **kwargs):
    return [kilde for _box, kilde, _conf in run(tokens, yolo_boxes, **kwargs)]


# ── Paddle boxes ─────────────────────────────────────────────

def test_box_covers_the_last_five_digits_only():
    token, = line(make_fnr())
    (box, mod11, _rec, _features), = sladd_boxes_from_tokens([token])
    x0, _y0, x1, _y1 = box
    sixth_digit_middle = token.x0 + 5.5 * CHAR_W
    seventh_digit_left = token.x0 + 6 * CHAR_W
    assert mod11
    assert sixth_digit_middle <= x0 <= seventh_digit_left
    assert x1 >= token.x1


def test_box_when_ocr_splits_date_and_personnummer():
    fnr = make_fnr()
    first, second = line(fnr[:6], fnr[6:])
    (box, _mod11, _rec, _features), = sladd_boxes_from_tokens([first, second])
    assert first.x1 - CHAR_W / 2 <= box[0] <= second.x0
    assert box[2] >= second.x1


def test_paddle_box_is_kept_without_yolo():
    assert kilder(line("Eier:", make_fnr())) == ["paddle"]


def test_no_box_for_a_number_with_a_wrong_check_digit():
    fnr = make_fnr()
    assert kilder(line(fnr[:-1] + str((int(fnr[-1]) + 1) % 10))) == []


def test_paddle_box_stitched_across_a_column_gap_is_dropped():
    # The line text has one space between tokens whatever the distance, so
    # numbers in two table columns can read as one fnr.
    # The limit is RULE_WINDOW max_gap, 6.5 digit widths. The tokens are tall
    # so the long box still passes the Paddle shape filter.
    fnr = make_fnr()
    first, = line(fnr[:7], height=70)
    second, = line(fnr[7:], x=first.x1 + 7 * CHAR_W, height=70)
    (box, _kilde, _conf), = run([first, second], postfilter=False)
    assert not has_paddle_noise_shape(box)
    assert kilder([first, second]) == []
    close, = line(fnr[7:], x=first.x1 + CHAR_W, height=70)
    assert kilder([first, close]) == ["paddle"]


def test_paddle_box_stitched_across_a_decimal_point_is_dropped():
    fnr = make_fnr()
    assert kilder(line(f"{fnr[:7]}.{fnr[7:]}")) == []


# ── YOLO boxes ───────────────────────────────────────────────

def test_yolo_over_a_paddle_box_makes_it_begge():
    tokens = line(make_fnr())
    paddle_box, = [box for box, _k, _c in run(tokens)]
    (box, kilde, conf), = run(tokens, [(*paddle_box, 0.8)])
    assert (box, kilde, conf) == (paddle_box, "begge", 0.8)


def test_yolo_over_a_word_is_rejected():
    word, = line("Eiendom")
    assert kilder([word], [(*span(word, 1, 6), 0.9)]) == []


def test_yolo_without_text_needs_some_confidence():
    empty_area = (200, 1000, 400, 1050)
    assert kilder([], [(*empty_area, 0.01)]) == []
    assert kilder([], [(*empty_area, 0.3)]) == ["yolo"]


def test_yolo_over_a_coordinate_follows_the_decimal_rules():
    # A coordinate has a decimal point, an fnr never does. Each tier applies
    # only when OCR read the box well and YOLO was not confident.
    def coordinate(rec, conf):
        token, = line("6626630.58", rec=rec)
        return kilder([token], [(*span(token, 3, 9), conf)])

    main, low = RULE_DECIMAL, RULE_DECIMAL_LOW_TIER
    between_tiers = (low["ocr_conf_exempt"] + main["ocr_conf_exempt"]) / 2
    low_tier_rec = (low["rec_veto"] + main["rec_veto"]) / 2
    below_both = low["ocr_conf_exempt"] - 0.1

    assert coordinate(rec=0.999, conf=between_tiers) == []
    assert coordinate(rec=low_tier_rec, conf=below_both) == []
    assert coordinate(rec=low_tier_rec, conf=between_tiers) == ["yolo"]
    assert coordinate(rec=0.999, conf=main["ocr_conf_exempt"]) == ["yolo"]
    assert coordinate(rec=low["rec_veto"] - 0.05, conf=below_both) == ["yolo"]


def test_koordfam_drops_yolo_without_an_fnr_candidate():
    token, = line("12345")
    yolo = [(token.x0, token.y0, token.x1, token.y1, 0.3)]
    assert kilder([token], yolo) == ["yolo"]
    assert kilder([token], yolo, koordfam=True) == []


def test_koordfam_keeps_paddle_fnr():
    assert kilder(line("Eier:", make_fnr()), koordfam=True) == ["paddle"]


def test_too_small_yolo_box_is_dropped():
    token, = line("12345")
    tiny = (token.x0, token.y0, token.x0 + 20, token.y0 + 20, 0.9)
    assert kilder([token], [tiny]) == []


def test_square_yolo_box_needs_high_confidence():
    # Too square for a row of digits. A confident YOLO box skips that check.
    token, = line("12345")
    square = (token.x0 + 20, token.y0 - 20, token.x0 + 120, token.y1 + 20)
    assert kilder([token], [(*square, YOLO_CONF_GEOMETRY_THRESHOLD - 0.1)]) == []
    assert kilder([token], [(*square, YOLO_CONF_GEOMETRY_THRESHOLD)]) == ["yolo"]


def test_postfilter_off_keeps_the_rule_rejections():
    token, = line("6626630.58")
    yolo = [(*span(token, 3, 9), 0.3)]
    assert kilder([token], yolo) == []
    assert kilder([token], yolo, postfilter=False) == ["yolo"]
