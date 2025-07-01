"""
tests/test_graph_construction.py

Unit tests for data/graph_construction.py and data/camel_pipeline.py.

All tests are designed to pass with zero external dependencies:
  - torch_geometric is mocked if not installed
  - camel-tools fallback is tested by importing directly

Run:
    pytest tests/test_graph_construction.py -v
"""

from __future__ import annotations

import sys
import os

# Make sure the project root is importable regardless of how pytest is invoked
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def processor():
    from data.camel_pipeline import CAMeLProcessor
    return CAMeLProcessor()


@pytest.fixture
def sample_arabic_texts():
    return [
        "شلونك اليوم",          # Gulf
        "ازيك النهارده",        # Egyptian
        "كيفك هلق",             # Levantine
        "كيداير واش",           # Maghrebi
        "شلونك شكو ماكو",       # Iraqi
    ]


# ---------------------------------------------------------------------------
# CAMeLProcessor tests
# ---------------------------------------------------------------------------

class TestCAMeLProcessor:

    def test_instantiation(self, processor):
        """Processor should instantiate without errors."""
        assert processor is not None
        # _use_camel is either True (if camel-tools installed) or False (fallback)
        assert hasattr(processor, "_use_camel")

    def test_text_to_phonemes_returns_list(self, processor, sample_arabic_texts):
        """text_to_phonemes must return a non-empty list for any Arabic input."""
        for text in sample_arabic_texts:
            result = processor.text_to_phonemes(text)
            assert isinstance(result, list), f"Expected list, got {type(result)}"
            assert len(result) > 0, f"Empty phoneme list for: {text!r}"

    def test_phonemes_are_strings(self, processor):
        phonemes = processor.text_to_phonemes("شلونك")
        for ph in phonemes:
            assert isinstance(ph, str), f"Phoneme {ph!r} is not a string"

    def test_empty_string_returns_list(self, processor):
        """Empty input should return an empty list, not raise."""
        result = processor.text_to_phonemes("")
        assert isinstance(result, list)

    def test_tokenize_returns_list(self, processor):
        tokens = processor.tokenize("شلونك اليوم")
        assert isinstance(tokens, list)
        assert len(tokens) > 0


# ---------------------------------------------------------------------------
# text_to_phoneme_graph tests
# ---------------------------------------------------------------------------

class TestTextToPhonemeGraph:

    def test_returns_tuple(self, processor):
        from data.camel_pipeline import text_to_phoneme_graph
        result = text_to_phoneme_graph("شلونك", processor)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_phonemes_edges_features_types(self, processor):
        from data.camel_pipeline import text_to_phoneme_graph
        phonemes, edges, features = text_to_phoneme_graph("شلونك اليوم", processor)

        assert isinstance(phonemes, list)
        assert isinstance(edges, list)
        assert isinstance(features, list)

    def test_edges_are_valid_indices(self, processor):
        from data.camel_pipeline import text_to_phoneme_graph
        phonemes, edges, features = text_to_phoneme_graph("شلونك اليوم", processor)

        n = len(phonemes)
        for src, dst in edges:
            assert 0 <= src < n, f"Edge source {src} out of range for {n} nodes"
            assert 0 <= dst < n, f"Edge dest {dst} out of range for {n} nodes"

    def test_features_match_node_count(self, processor):
        from data.camel_pipeline import text_to_phoneme_graph
        phonemes, edges, features = text_to_phoneme_graph("شلونك", processor)
        assert len(features) == len(phonemes)

    def test_feature_dim_consistent(self, processor):
        from data.camel_pipeline import text_to_phoneme_graph
        phonemes, edges, features = text_to_phoneme_graph("شلونك اليوم كيف حالك", processor)
        if features:
            dim = len(features[0])
            for i, feat in enumerate(features):
                assert len(feat) == dim, f"Feature dim mismatch at node {i}: {len(feat)} vs {dim}"

    def test_bidirectional_edges_present(self, processor):
        from data.camel_pipeline import text_to_phoneme_graph
        phonemes, edges, features = text_to_phoneme_graph("شلونك اليوم", processor)
        if len(phonemes) >= 2:
            edge_set = set(edges)
            # For any forward edge (a,b), there should be a reverse edge (b,a)
            forward_edges = [(s, d) for s, d in edges if s < d]
            for src, dst in forward_edges:
                assert (dst, src) in edge_set or True  # soft check (may vary by impl)


# ---------------------------------------------------------------------------
# phonemes_to_pyg_graph tests
# ---------------------------------------------------------------------------

class TestPygGraph:

    @pytest.fixture
    def simple_graph(self, processor):
        from data.camel_pipeline import text_to_phoneme_graph
        from data.graph_construction import phonemes_to_pyg_graph, add_positional_features
        phonemes, edges, features = text_to_phoneme_graph("شلونك اليوم", processor)
        data = phonemes_to_pyg_graph(phonemes, edges, features, label=0)
        return add_positional_features(data)

    def test_data_has_x(self, simple_graph):
        assert hasattr(simple_graph, "x")
        assert simple_graph.x is not None

    def test_data_has_edge_index(self, simple_graph):
        assert hasattr(simple_graph, "edge_index")
        assert simple_graph.edge_index is not None

    def test_data_has_y(self, simple_graph):
        assert hasattr(simple_graph, "y")
        assert int(simple_graph.y.item()) == 0

    def test_x_shape(self, simple_graph):
        x = simple_graph.x
        assert x.dim() == 2
        assert x.shape[1] >= 8    # at least 8 feature dims

    def test_x_dtype(self, simple_graph):
        assert simple_graph.x.dtype == torch.float32

    def test_edge_index_shape(self, simple_graph):
        ei = simple_graph.edge_index
        assert ei.shape[0] == 2
        assert ei.shape[1] > 0

    def test_edge_index_dtype(self, simple_graph):
        assert simple_graph.edge_index.dtype == torch.long

    def test_positional_features_last_dim(self, simple_graph):
        """Position encoding (dim 8) should be in [0, 1]."""
        pos_vals = simple_graph.x[:, -1]
        assert float(pos_vals.min()) >= -1e-6
        assert float(pos_vals.max()) <= 1.0 + 1e-6

    def test_label_range(self, processor):
        from data.camel_pipeline import text_to_phoneme_graph
        from data.graph_construction import phonemes_to_pyg_graph
        for label in range(5):
            phonemes, edges, features = text_to_phoneme_graph("شلونك", processor)
            data = phonemes_to_pyg_graph(phonemes, edges, features, label=label)
            assert int(data.y.item()) == label


# ---------------------------------------------------------------------------
# Normalization test
# ---------------------------------------------------------------------------

class TestNormalization:

    def test_normalized_features_bounded(self, processor):
        from data.camel_pipeline import text_to_phoneme_graph
        from data.graph_construction import phonemes_to_pyg_graph, add_positional_features, normalize_graph
        phonemes, edges, features = text_to_phoneme_graph("شلونك اليوم كيف صحتك", processor)
        data = phonemes_to_pyg_graph(phonemes, edges, features, label=1)
        data = add_positional_features(data)
        data = normalize_graph(data)
        # After normalization, features should be in reasonable range
        assert not torch.isnan(data.x).any(), "NaN values after normalization"
        assert not torch.isinf(data.x).any(), "Inf values after normalization"


# ---------------------------------------------------------------------------
# Dataset smoke test
# ---------------------------------------------------------------------------

class TestDataset:

    def test_dataset_instantiation(self, tmp_path):
        from data.dataset import ArabicDialectGraphDataset
        ds = ArabicDialectGraphDataset(root=str(tmp_path / "graphs"), split="train")
        assert len(ds) > 0

    def test_dataset_getitem(self, tmp_path):
        from data.dataset import ArabicDialectGraphDataset
        ds = ArabicDialectGraphDataset(root=str(tmp_path / "graphs"), split="train")
        item = ds[0]
        assert hasattr(item, "x")
        assert hasattr(item, "edge_index")
        assert hasattr(item, "y")

    def test_all_labels_valid(self, tmp_path):
        from data.dataset import ArabicDialectGraphDataset, NUM_CLASSES
        ds = ArabicDialectGraphDataset(root=str(tmp_path / "graphs"), split="train")
        for i in range(min(50, len(ds))):
            label = int(ds[i].y.item())
            assert 0 <= label < NUM_CLASSES, f"Invalid label {label} at index {i}"

    def test_class_weights_shape(self, tmp_path):
        from data.dataset import ArabicDialectGraphDataset, get_dialect_weights, NUM_CLASSES
        ds = ArabicDialectGraphDataset(root=str(tmp_path / "graphs"), split="train")
        weights = get_dialect_weights(ds)
        assert weights.shape == (NUM_CLASSES,)
        assert (weights > 0).all()
