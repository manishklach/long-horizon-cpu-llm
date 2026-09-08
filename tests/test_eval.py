"""Unit tests for the eval harness (no model download needed)."""
import random
from src.eval.records import (
    generate_store, recall_item, missing_item,
    score_recall, score_abstain, ABSTAIN_PHRASE, ABSENT_POOL)


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


def test_scorers():
    assert score_recall("The office is Berlin.", "berlin")
    assert not score_recall("The office is Lima.", "Berlin")
    assert score_abstain(f"Sorry, {ABSTAIN_PHRASE}.")
    assert not score_abstain("The office is Berlin.")
