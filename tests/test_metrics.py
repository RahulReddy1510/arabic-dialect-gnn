"""
tests/test_metrics.py

Unit tests for evaluation/metrics.py.

All tests use synthetic preds/labels so there are
zero external dependencies.

Run:
    pytest tests/test_metrics.py -v
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np

from evaluation.metrics import (
    compute_macro_f1,
    compute_per_class_f1,
    compute_confusion_matrix_normalized,
    compute_classification_report,
    bootstrap_f1_confidence_interval,
    compute_per_length_f1,
    print_metrics_summary,
    DIALECT_NAMES,
)

NUM_CLASSES = len(DIALECT_NAMES)
rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Synthetic prediction sets
# ---------------------------------------------------------------------------

def perfect_preds(n=100):
    labels = list(range(NUM_CLASSES)) * (n // NUM_CLASSES)
    return labels, labels[:]

def random_preds(n=200, seed=42):
    rng_ = np.random.default_rng(seed)
    labels = rng_.integers(0, NUM_CLASSES, size=n).tolist()
    preds  = rng_.integers(0, NUM_CLASSES, size=n).tolist()
    return preds, labels

def biased_preds(n=200):
    """Biased toward class 0 — simulates Gulf under-prediction."""
    labels = (list(range(NUM_CLASSES)) * (n // NUM_CLASSES))[:n]
    preds  = [0 if l == 0 else rng.integers(1, NUM_CLASSES) for l in labels]
    return preds, labels


# ---------------------------------------------------------------------------
# compute_macro_f1
# ---------------------------------------------------------------------------

class TestComputeMacroF1:

    def test_perfect_f1(self):
        labels, preds = perfect_preds()
        f1 = compute_macro_f1(preds, labels)
        assert abs(f1 - 1.0) < 1e-6

    def test_random_f1_in_range(self):
        preds, labels = random_preds()
        f1 = compute_macro_f1(preds, labels)
        assert 0.0 <= f1 <= 1.0

    def test_returns_float(self):
        preds, labels = random_preds()
        f1 = compute_macro_f1(preds, labels)
        assert isinstance(f1, float)

    def test_worse_than_perfect(self):
        preds, labels = random_preds()
        f1 = compute_macro_f1(preds, labels)
        assert f1 < 1.0


# ---------------------------------------------------------------------------
# compute_per_class_f1
# ---------------------------------------------------------------------------

class TestComputePerClassF1:

    def test_keys_are_dialect_names(self):
        preds, labels = random_preds()
        result = compute_per_class_f1(preds, labels)
        assert set(result.keys()) == set(DIALECT_NAMES)

    def test_values_in_range(self):
        preds, labels = random_preds()
        result = compute_per_class_f1(preds, labels)
        for dialect, f1 in result.items():
            assert 0.0 <= f1 <= 1.0, f"{dialect}: F1 {f1} out of [0,1]"

    def test_perfect_per_class(self):
        labels, preds = perfect_preds()
        result = compute_per_class_f1(preds, labels)
        for dialect, f1 in result.items():
            assert abs(f1 - 1.0) < 1e-6, f"{dialect}: expected 1.0, got {f1}"

    def test_biased_preds_low_some_classes(self):
        preds, labels = biased_preds()
        result = compute_per_class_f1(preds, labels)
        # Non-zero F1 for at least one class
        assert any(f > 0 for f in result.values())


# ---------------------------------------------------------------------------
# compute_confusion_matrix_normalized
# ---------------------------------------------------------------------------

class TestConfusionMatrix:

    def test_shape(self):
        preds, labels = random_preds()
        cm = compute_confusion_matrix_normalized(preds, labels)
        assert cm.shape == (NUM_CLASSES, NUM_CLASSES)

    def test_rows_sum_to_one(self):
        preds, labels = random_preds(n=500)
        cm = compute_confusion_matrix_normalized(preds, labels)
        row_sums = cm.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(NUM_CLASSES), atol=1e-6)

    def test_values_in_zero_one(self):
        preds, labels = random_preds()
        cm = compute_confusion_matrix_normalized(preds, labels)
        assert (cm >= 0).all()
        assert (cm <= 1 + 1e-9).all()

    def test_perfect_preds_diagonal(self):
        labels, preds = perfect_preds()
        cm = compute_confusion_matrix_normalized(preds, labels)
        np.testing.assert_allclose(np.diag(cm), np.ones(NUM_CLASSES), atol=1e-6)


# ---------------------------------------------------------------------------
# bootstrap_f1_confidence_interval
# ---------------------------------------------------------------------------

class TestBootstrapCI:

    def test_returns_tuple_of_two_floats(self):
        preds, labels = random_preds()
        lower, upper = bootstrap_f1_confidence_interval(preds, labels, n_bootstrap=100)
        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_lower_le_upper(self):
        preds, labels = random_preds()
        lower, upper = bootstrap_f1_confidence_interval(preds, labels, n_bootstrap=200)
        assert lower <= upper

    def test_ci_contains_point_estimate(self):
        preds, labels = random_preds(n=500)
        point = compute_macro_f1(preds, labels)
        lower, upper = bootstrap_f1_confidence_interval(preds, labels, n_bootstrap=500)
        # Very unlikely to fail if bootstrap is implemented correctly
        assert lower <= point + 0.02
        assert upper >= point - 0.02

    def test_ci_in_range(self):
        preds, labels = random_preds()
        lower, upper = bootstrap_f1_confidence_interval(preds, labels, n_bootstrap=100)
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0


# ---------------------------------------------------------------------------
# compute_per_length_f1
# ---------------------------------------------------------------------------

class TestPerLengthF1:

    def test_returns_dict(self):
        preds, labels = random_preds(n=200)
        lengths = rng.integers(1, 30, size=200).tolist()
        result = compute_per_length_f1(preds, labels, lengths)
        assert isinstance(result, dict)

    def test_all_bins_present(self):
        preds, labels = random_preds(n=200)
        lengths = rng.integers(1, 30, size=200).tolist()
        bins = [0, 5, 10, 20, int(1e9)]
        result = compute_per_length_f1(preds, labels, lengths, bins=bins)
        assert len(result) == len(bins) - 1

    def test_f1_values_in_range(self):
        preds, labels = random_preds(n=400)
        lengths = rng.integers(1, 30, size=400).tolist()
        result = compute_per_length_f1(preds, labels, lengths)
        for bin_label, stats in result.items():
            if stats["macro_f1"] is not None:
                assert 0.0 <= stats["macro_f1"] <= 1.0, f"{bin_label}: {stats['macro_f1']}"

    def test_empty_bin_returns_none(self):
        """A bin that contains no samples should return None for macro_f1."""
        preds  = [0, 1, 2, 3, 4]
        labels = [0, 1, 2, 3, 4]
        lengths = [3, 3, 3, 3, 3]   # All in bin [0, 5)
        result = compute_per_length_f1(preds, labels, lengths, bins=[0, 5, 10, int(1e9)])
        # Bins [5,10) and [10,inf) should have no samples
        for _, stats in list(result.items())[1:]:
            assert stats["n_samples"] == 0


# ---------------------------------------------------------------------------
# compute_classification_report
# ---------------------------------------------------------------------------

class TestClassificationReport:

    def test_returns_string(self):
        preds, labels = random_preds()
        report = compute_classification_report(preds, labels)
        assert isinstance(report, str)

    def test_contains_dialect_names(self):
        preds, labels = random_preds()
        report = compute_classification_report(preds, labels)
        for dialect in DIALECT_NAMES:
            assert dialect in report, f"{dialect} not found in report"


# ---------------------------------------------------------------------------
# print_metrics_summary (smoke test)
# ---------------------------------------------------------------------------

class TestPrintMetricsSummary:

    def test_does_not_raise(self, capsys):
        preds, labels = random_preds()
        print_metrics_summary("TestModel", preds, labels)
        captured = capsys.readouterr()
        assert "Macro F1" in captured.out
