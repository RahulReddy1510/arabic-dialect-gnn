"""
tests/test_model.py

Unit tests for the DialectGAT model.

Tests cover:
  - Model instantiation with default and custom configs
  - Forward pass output shapes
  - Attention weight return
  - Checkpoint save/load round-trip
  - build_gat factory function
  - num_parameters sanity check

Run:
    pytest tests/test_model.py -v
"""

from __future__ import annotations

import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
from torch_geometric.data import Data

from models.gat_model import DialectGAT, GATLayer, build_gat, DIALECT_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(n_graphs: int = 2, n_nodes_each: int = 8, feat_dim: int = 9) -> Data:
    """Create a synthetic mini-batch of phoneme graphs."""
    total_nodes = n_graphs * n_nodes_each
    x = torch.randn(total_nodes, feat_dim)

    # Build edges within each graph (sequential + skip-2)
    edge_srcs, edge_dsts = [], []
    for g in range(n_graphs):
        offset = g * n_nodes_each
        for i in range(n_nodes_each - 1):
            edge_srcs += [offset + i, offset + i + 1]
            edge_dsts += [offset + i + 1, offset + i]
        for i in range(n_nodes_each - 2):
            edge_srcs += [offset + i, offset + i + 2]
            edge_dsts += [offset + i + 2, offset + i]

    edge_index = torch.tensor([edge_srcs, edge_dsts], dtype=torch.long)
    batch = torch.repeat_interleave(
        torch.arange(n_graphs), torch.tensor([n_nodes_each] * n_graphs)
    )
    y = torch.randint(0, 5, (n_graphs,))

    data = Data(x=x, edge_index=edge_index, batch=batch, y=y)
    return data


# ---------------------------------------------------------------------------
# DialectGAT tests
# ---------------------------------------------------------------------------

class TestDialectGAT:

    def test_default_instantiation(self):
        model = DialectGAT()
        assert model is not None

    def test_custom_config(self):
        model = DialectGAT(
            node_feature_dim=9,
            num_classes=5,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            pooling="mean",
        )
        assert model is not None

    def test_num_parameters_positive(self):
        model = DialectGAT()
        n_params = model.num_parameters()
        assert n_params > 0

    def test_num_parameters_reasonable(self):
        """Default config should be well under 1M parameters."""
        model = DialectGAT(hidden_dim=128, num_heads=8, num_layers=3)
        n_params = model.num_parameters()
        assert n_params < 1_000_000, f"Too many parameters: {n_params:,}"

    def test_forward_logits_shape(self):
        model = DialectGAT()
        model.eval()
        batch = _make_batch(n_graphs=4)
        with torch.no_grad():
            logits, attn = model(batch)
        assert logits.shape == (4, 5), f"Expected (4,5), got {logits.shape}"

    def test_forward_attn_weights_count(self):
        """forward() must return one attention tensor per GAT layer."""
        n_layers = 3
        model = DialectGAT(num_layers=n_layers)
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            _, attn = model(batch)
        assert len(attn) == n_layers, f"Expected {n_layers} attn tensors, got {len(attn)}"

    def test_attn_weights_shape(self):
        """Attention weights should be (E_with_self_loops, num_heads)."""
        model = DialectGAT(num_heads=8)
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            _, attn = model(batch)
        for i, alpha in enumerate(attn):
            assert alpha.dim() == 2, f"Layer {i} attn should be 2D"
            assert alpha.shape[1] == 8, f"Layer {i} should have 8 heads"

    def test_output_no_nan(self):
        model = DialectGAT()
        model.eval()
        batch = _make_batch(n_graphs=8)
        with torch.no_grad():
            logits, _ = model(batch)
        assert not torch.isnan(logits).any(), "NaN in logits"
        assert not torch.isinf(logits).any(), "Inf in logits"

    def test_pooling_mean(self):
        model = DialectGAT(pooling="mean")
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            logits, _ = model(batch)
        assert logits.shape[0] == 2

    def test_pooling_max(self):
        model = DialectGAT(pooling="max")
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            logits, _ = model(batch)
        assert logits.shape[0] == 2

    def test_hidden_dim_must_be_divisible_by_heads(self):
        with pytest.raises(AssertionError):
            DialectGAT(hidden_dim=100, num_heads=8)  # 100 % 8 != 0

    def test_single_graph_inference(self):
        model = DialectGAT()
        model.eval()
        # Single graph without batch vector
        x = torch.randn(6, 9)
        edge_index = torch.tensor([[0,1,1,2,2,3,3,4,4,5], [1,0,2,1,3,2,4,3,5,4]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, batch=None)
        with torch.no_grad():
            logits, attn = model(data)
        assert logits.shape == (1, 5)

    def test_predict_returns_dialect_name_and_confidence(self):
        model = DialectGAT()
        x = torch.randn(6, 9)
        edge_index = torch.tensor([[0,1,1,2,2,3,3,4,4,5], [1,0,2,1,3,2,4,3,5,4]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, batch=None)
        dialect, conf = model.predict(data)
        assert dialect in DIALECT_NAMES
        assert 0.0 <= conf <= 1.0

    def test_checkpoint_save_load_roundtrip(self):
        model = DialectGAT(hidden_dim=64, num_heads=4, num_layers=2)
        model.eval()

        batch = _make_batch()
        with torch.no_grad():
            logits_before, _ = model(batch)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_ckpt.pth")
            model.save_checkpoint(ckpt_path)
            restored = DialectGAT.from_pretrained(ckpt_path)
            restored.eval()
            with torch.no_grad():
                logits_after, _ = restored(batch)

        assert torch.allclose(logits_before, logits_after, atol=1e-5), \
            "Logits changed after checkpoint round-trip"

    def test_training_step_does_not_crash(self):
        """A single forward+backward should not raise."""
        model = DialectGAT(hidden_dim=64, num_heads=4, num_layers=2, dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = _make_batch(n_graphs=4)
        model.train()
        optimizer.zero_grad()
        logits, _ = model(batch)
        loss = torch.nn.functional.cross_entropy(logits, batch.y)
        loss.backward()
        optimizer.step()
        assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# GATLayer tests
# ---------------------------------------------------------------------------

class TestGATLayer:

    def test_output_shape_concat(self):
        layer = GATLayer(in_channels=32, out_channels=4, heads=8, concat=True)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0,1,1,2,2,3,3,4], [1,0,2,1,3,2,4,3]], dtype=torch.long)
        x_out, alpha = layer(x, edge_index)
        assert x_out.shape == (10, 32)  # 4 * 8 = 32

    def test_attention_weights_returned(self):
        layer = GATLayer(in_channels=32, out_channels=4, heads=8)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0,1,1,2,2,3,3,4], [1,0,2,1,3,2,4,3]], dtype=torch.long)
        x_out, alpha = layer(x, edge_index)
        assert alpha is not None
        assert alpha.dim() == 2
        assert alpha.shape[1] == 8


# ---------------------------------------------------------------------------
# build_gat factory tests
# ---------------------------------------------------------------------------

class TestBuildGAT:

    def test_default_config(self):
        model = build_gat({})
        assert isinstance(model, DialectGAT)
        assert model.num_heads == 8
        assert model.num_layers == 3
        assert model.hidden_dim == 128

    def test_custom_config_nested(self):
        config = {"model": {"hidden_dim": 64, "num_heads": 4, "num_layers": 2}}
        model = build_gat(config)
        assert model.hidden_dim == 64
        assert model.num_heads == 4

    def test_custom_config_flat(self):
        config = {"hidden_dim": 64, "num_heads": 4}
        model = build_gat(config)
        assert model.hidden_dim == 64
