"""Unit tests for data splitting determinism."""

from src.prepare_data import stable_bucket


def test_stable_bucket_determinism():
    # Same tweet ID must always hash to the same bucket
    assert stable_bucket(123456789) == stable_bucket(123456789)
    assert 0 <= stable_bucket(123456789) < 100
    assert 0 <= stable_bucket(987654321) < 100
