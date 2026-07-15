"""
Unit tests for agents/_affordance.py -- the shared deterministic
"does human use suit this product?" signal behind every product-conditional
human-centric bias (Concept Agent prompt/re-prompt, Hook-Checker tiebreak,
Shot-List Agent zero-human-shot retry). Deterministic pure function, so these
are direct, no-network tests.
"""
from __future__ import annotations

from agents._affordance import human_use_suits_product


def _truth(fact: str, category: str = "construction_detail") -> dict:
    return {"truth_id": "t1", "fact": fact, "category": category, "source": "photo_1"}


def test_contact_part_fact_establishes_affordance():
    truths = [_truth("adjustable shoulder strap with a stitched pad")]
    assert human_use_suits_product(truths) is True


def test_plural_contact_part_matches_via_plural_strip():
    truths = [_truth("twin padded handles stitched to the top panel")]
    assert human_use_suits_product(truths) is True


def test_body_scale_form_factor_fact_establishes_affordance():
    truths = [_truth("a rounded case that sits in the palm of a hand", category="form_factor")]
    assert human_use_suits_product(truths) is True


def test_body_word_in_non_scale_category_does_not_count():
    # "hand-sized dent" mentions a body word, but an imperfection fact isn't a
    # statement about the object's overall scale -- must not trigger.
    truths = [_truth("a hand shaped dent on the rear housing", category="imperfection")]
    assert human_use_suits_product(truths) is False


def test_plain_still_life_product_has_no_affordance():
    truths = [
        _truth("matte black anodized aluminum housing", category="material"),
        _truth("dual cylindrical hinge with knurled end caps"),
        _truth("faint scuff on the base plate cutout", category="imperfection"),
    ]
    assert human_use_suits_product(truths) is False


def test_empty_and_missing_truths_are_false():
    assert human_use_suits_product([]) is False
    assert human_use_suits_product(None) is False


# ---------------------------------------------------------------------------
# Kitchen/hand-use goods — added 2026-07-15.
# ---------------------------------------------------------------------------

def test_pan_fact_establishes_human_use_affordance():
    truths = [_truth("compact cast-iron pan with helper handle", category="form_factor")]
    assert human_use_suits_product(truths) is True


def test_knife_fact_establishes_human_use_affordance():
    truths = [_truth("8-inch chef's knife with full-tang blade and riveted handle")]
    assert human_use_suits_product(truths) is True


def test_cutting_board_fact_establishes_affordance():
    truths = [_truth("end-grain walnut cutting board with rubber feet", category="form_factor")]
    assert human_use_suits_product(truths) is True


def test_whisk_establishes_affordance():
    truths = [_truth("stainless steel balloon whisk with silicone-grip handle")]
    assert human_use_suits_product(truths) is True


def test_held_in_scale_fact_establishes_affordance():
    """Action verb 'held' in a scale_cue fact triggers body-scale check."""
    truths = [_truth("easily held in one hand during pouring", category="scale_cue")]
    assert human_use_suits_product(truths) is True


def test_poured_in_form_factor_establishes_affordance():
    truths = [_truth("32-oz pitcher poured directly over ice without dripping", category="form_factor")]
    assert human_use_suits_product(truths) is True


def test_decorative_vase_has_no_affordance():
    """A decorative product with no carry/contact parts returns False."""
    truths = [
        _truth("tall cylindrical ceramic vase with blue glaze finish", category="form_factor"),
        _truth("matte cobalt blue exterior with glossy interior", category="color"),
    ]
    assert human_use_suits_product(truths) is False
