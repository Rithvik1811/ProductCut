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
