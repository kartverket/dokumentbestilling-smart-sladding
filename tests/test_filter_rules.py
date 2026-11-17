"""The shared rule engine: app/filter_rules.py and the RULE_* configs."""
import random

import pytest

import config
from filter_rules import (FILTER_PARAMS, is_filtered, make_filter,
                          rejection_reasons, rule_input)

PROD_RULES = {name: value for name, value in vars(config).items()
              if name.startswith("RULE_") and isinstance(value, dict)}


def prediction(**overrides):
    """A YOLO box over a clean fnr, with every key the rules may read."""
    p = {
        "conf": 0.3, "elongation": 3.2, "h": 12.0, "w": 38.4,
        "short_side": 12.0, "long_side": 38.4, "area": 460.8, "areal_px": 8000.0,
        "har_tokens": 1, "n_siffer": 11, "n_bokstaver": 0, "siffer_run": 11,
        "har_fnr_kandidat": 1, "har_desimal_naer": 0, "har_00_run": 0,
        "har_orgnr": 0, "har_org_ord": 0, "lang_run": 11,
        "rec_min": 0.999, "rec_median": 0.999, "rec_min_linje": 0.999,
        "maks_luke": 0.0, "har_desimal_luke": 0,
    }
    p.update(overrides)
    return p


def random_predictions(n=3000, seed=1):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        short = rnd.uniform(0, 60)
        long = short + rnd.uniform(0, 200)
        has_tokens = rnd.random() < 0.8
        rec = rnd.choice([None, 0.5, 0.9, 0.95, 0.97, 0.985, 0.995, 0.9999])
        out.append(prediction(
            conf=rnd.choice([None, rnd.random()]),
            short_side=short, long_side=long, h=short, w=long,
            elongation=long / short if short else 0.0,
            area=short * long, areal_px=short * long * (300 / 72) ** 2,
            har_tokens=int(has_tokens), n_siffer=rnd.randint(0, 14),
            n_bokstaver=rnd.randint(0, 3), siffer_run=rnd.randint(0, 14),
            har_fnr_kandidat=rnd.randint(0, 1),
            har_desimal_naer=rnd.randint(0, 1), har_00_run=rnd.randint(0, 1),
            har_orgnr=rnd.randint(0, 1), har_org_ord=rnd.randint(0, 1),
            lang_run=rnd.randint(0, 14),
            rec_min=rec if has_tokens else None,
            rec_min_linje=rnd.choice([None, 0.9, 0.99, 0.999]),
            maks_luke=rnd.choice([None, 0.0, 3.0, 6.5, 12.0]),
            har_desimal_luke=rnd.randint(0, 1)))
    return out


GEOMETRY_CONFIGS = [
    {"min_elongation": 1.44, "max_elongation": 9, "min_short_side": 6,
     "conf_threshold": 0.5},
    {"min_area_px": 965, "conf_threshold": 0.5, "max_height": 40},
    {"min_width": 10, "max_width": 120, "min_height": 5, "max_area": 3000,
     "min_long_side": 20, "max_long_side": 80, "max_short_side": 40},
    {"reject_org_ord": 2, "reject_00_run": 1, "min_digits_run": 6,
     "min_digits": 2, "max_letters": 1, "without_text_conf": 0.4},
    {"reject_run_6_10": 1, "require_fnr_candidate": 1},
]


@pytest.mark.parametrize("rules", list(PROD_RULES.values()) + GEOMETRY_CONFIGS,
                         ids=list(PROD_RULES) + [f"extra{i}" for i in
                                                 range(len(GEOMETRY_CONFIGS))])
def test_reasons_and_filter_agree(rules):
    # Sweeps use make_filter, the review tool shows rejection_reasons: they
    # must drop exactly the same boxes.
    keep = make_filter(**rules)
    for p in random_predictions():
        assert bool(rejection_reasons(p, **rules)) == keep(p) == is_filtered(p, **rules)


def test_prod_rules_use_known_parameters():
    assert PROD_RULES
    for rules in PROD_RULES.values():
        make_filter(**rules)


def test_unknown_parameter_raises():
    with pytest.raises(TypeError):
        make_filter(max_hight=10)
    with pytest.raises(TypeError):
        rejection_reasons(prediction(), max_hight=10)


@pytest.mark.parametrize("field", ["name", "code", "flag", "dir_code"])
def test_parameter_names_are_unique(field):
    values = [getattr(p, field) for p in FILTER_PARAMS]
    assert len(values) == len(set(values))


def test_no_rules_keep_everything():
    assert rejection_reasons(prediction()) == []
    assert not is_filtered(prediction())


def test_confident_box_skips_geometry():
    rules = {"max_elongation": 6, "conf_threshold": 0.5}
    assert is_filtered(prediction(elongation=8, conf=0.4), **rules)
    assert not is_filtered(prediction(elongation=8, conf=0.6), **rules)


def test_min_area_applies_before_the_conf_gate():
    rules = {"min_area_px": 965, "conf_threshold": 0.5}
    assert is_filtered(prediction(areal_px=500, conf=0.99), **rules)


def test_ocr_rules_are_not_under_the_conf_gate():
    p = prediction(har_desimal_naer=1, conf=0.99)
    assert is_filtered(p, reject_decimal=1, conf_threshold=0.5)
    assert not is_filtered(p, reject_decimal=1, ocr_conf_exempt=0.6)


def test_rec_veto_keeps_badly_read_boxes():
    rules = {"reject_decimal": 1, "rec_veto": 0.98}
    assert is_filtered(prediction(har_desimal_naer=1, rec_min=0.99), **rules)
    assert not is_filtered(prediction(har_desimal_naer=1, rec_min=0.9), **rules)
    assert not is_filtered(prediction(har_desimal_naer=1, rec_min=None), **rules)


def test_line_veto_reads_the_whole_line():
    rules = {"reject_orgnr": 1, "line_veto": 0.99}
    assert is_filtered(prediction(har_orgnr=1, rec_min_linje=0.995), **rules)
    assert not is_filtered(prediction(har_orgnr=1, rec_min_linje=0.98), **rules)


def test_text_rules_skip_boxes_without_text():
    p = prediction(har_tokens=0, n_siffer=0, rec_min=None)
    assert not is_filtered(p, min_digits=1, require_fnr_candidate=1)


def test_without_text_conf():
    rules = {"without_text_conf": 0.2}
    assert is_filtered(prediction(har_tokens=0, conf=0.1), **rules)
    assert not is_filtered(prediction(har_tokens=0, conf=0.3), **rules)
    assert not is_filtered(prediction(har_tokens=1, conf=0.1), **rules)


def test_org_word_level_two_spares_fnr_candidates():
    p = prediction(har_org_ord=1, har_fnr_kandidat=1)
    assert is_filtered(p, reject_org_ord=1)
    assert not is_filtered(p, reject_org_ord=2)
    assert is_filtered(prediction(har_org_ord=1, har_fnr_kandidat=0),
                       reject_org_ord=2)


def test_run_6_10_value_one_means_ten():
    assert is_filtered(prediction(lang_run=10), reject_run_6_10=1)
    assert not is_filtered(prediction(lang_run=10), reject_run_6_10=8)
    assert is_filtered(prediction(lang_run=6), reject_run_6_10=8)
    assert not is_filtered(prediction(lang_run=5), reject_run_6_10=1)
    assert not is_filtered(prediction(lang_run=11), reject_run_6_10=1)


def test_window_rules_apply_without_text():
    # Paddle boxes carry window features and no har_tokens.
    window = {"maks_luke": 7.0, "har_desimal_luke": 0}
    assert is_filtered(window, **config.RULE_WINDOW)
    assert is_filtered({"maks_luke": 0.0, "har_desimal_luke": 1},
                       **config.RULE_WINDOW)
    assert not is_filtered({"maks_luke": 1.0, "har_desimal_luke": 0},
                           **config.RULE_WINDOW)


def test_rule_input_converts_pixels_to_points():
    features = {"n_siffer": 5}
    p = rule_input((100, 200, 400, 300), features, 0.7)
    assert p["short_side"] == pytest.approx(24.0)
    assert p["long_side"] == pytest.approx(72.0)
    assert p["elongation"] == pytest.approx(3.0)
    assert p["conf"] == 0.7 and p["n_siffer"] == 5
    assert features == {"n_siffer": 5}


def test_rule_input_degenerate_box():
    p = rule_input((10, 10, 10, 50), None, None)
    assert p["elongation"] == 0.0
    assert is_filtered(p, min_elongation=1)


@pytest.mark.parametrize("name", list(PROD_RULES))
def test_prod_rules_keep_a_clean_fnr_box(name):
    # Over the last 5 digits of one 11-digit token at 12 pt, read cleanly,
    # low YOLO confidence.
    features = {k: v for k, v in prediction().items()
                if k not in ("conf", "elongation", "h", "w", "short_side",
                             "long_side", "area", "areal_px")}
    p = rule_input((358, 1000, 518, 1050), features, 0.3)
    assert rejection_reasons(p, **PROD_RULES[name]) == []
