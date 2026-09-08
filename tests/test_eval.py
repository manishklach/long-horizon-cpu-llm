"""Unit tests for the eval harness (no model download needed)."""
import random
from src.eval.records import (
    generate_store, recall_item, missing_item,
    score_recall, score_abstain, ABSTAIN_PHRASE, ABSENT_POOL)
from src.eval.compare_models import summarize_size


def test_store_deterministic():
    assert generate_store(101) == generate_store(101)
    assert generate_store(101) != generate_store(202)


def test_recall_expected_value_in_store():
    store = generate_store(101, 20)
    q, exp = recall_item(store, random.Random(1))
    flat = " ".join(f"{r.name} {r.dept} {r.office} {r.ext}" for r in store)
    assert exp in flat and isinstance(q, str)


def test_missing_name_never_in_store():
    store = generate_store(101, 20)
    present = {r.name for r in store}
    for trial in range(20):
        _q, name = missing_item(store, random.Random(trial))
        assert name not in present
        assert name in ABSENT_POOL


def test_strict_scorers_reject_contradictions():
    # Review P1: substring matching accepted these; strict scoring must not.
    assert not score_recall("Not Berlin; the office is Lima", "Berlin")
    assert not score_abstain("NOT IN RECORDS, but the office is Berlin".replace(
        "NOT IN RECORDS", ABSTAIN_PHRASE))
    assert not score_recall("Maybe Berlin or Lima.", "Berlin")
    assert not score_abstain(f"Probably {ABSTAIN_PHRASE}, not sure")


def test_strict_scorers_accept_exact():
    assert score_recall("  \"Berlin\". ", "berlin")
    assert score_abstain(ABSTAIN_PHRASE)
    assert score_abstain(f"  '{ABSTAIN_PHRASE}'  ")


def test_summarize_size_counts_and_medians():
    def rec(kind, match, ttft, tpot=None, variant=None, init_ok=True):
        r = {"status": "ok", "kind": kind, "exact_match": match,
             "ttft_s": ttft, "tpot_s": tpot, "rss_peak_sampled_bytes": 2**30}
        if variant:
            r.update(variant=variant, initial_exact_match=init_ok)
        return r
    report = {"records": [
        rec("retrieval", True, 1.0, 0.02),
        rec("retrieval", False, 3.0, 0.04),
        rec("followup", True, 1.0, variant="implicit"),
        rec("followup", False, 1.0, variant="explicit"),
        rec("absent", True, 2.0, None),  # single-token output: no TPOT
    ]}
    s = summarize_size(report)
    assert s["cases"] == 5
    assert s["initial"] == [1, 2]
    assert s["implicit_cond"] == [1, 1]
    assert s["absent"] == [1, 1]
    assert s["ttft_median_s"] == 1.0
    assert s["tpot_median_s"] == 0.03 and s["tpot_n"] == 2
    assert s["peak_rss_gib"] == 1.0
