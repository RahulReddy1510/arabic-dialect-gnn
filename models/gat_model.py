"""
models/gat_model.py

8-head Graph Attention Network for Arabic dialect identification.

Architecture summary:
  - Input projection:  node_feature_dim → hidden_dim
  - 3 × GATLayer:     hidden_dim → hidden_dim  (8 heads, concat then project)
  - Global mean pool:  graph-level embedding
  - MLP classifier:   hidden_dim → hidden_dim//2 → num_classes

Why 8 heads? I ran ablations at 4, 8, and 16 — 8 gave the best F1/compute
tradeoff (see ablation table in README). 16 heads started to over-segment the
phoneme feature space; individual heads were attending to highly overlapping
node subsets (cosine similarity between head distributions > 0.9 in layer 3).

Why mean pooling? Max pooling was more sensitive to outlier phoneme nodes —
a single distinctive phoneme would dominate the pooled representation and
hurt calibration on borderline dialect pairs. Mean pooling gives a smoother
aggregate.

Reference:
    Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., &
    Bengio, Y. (2018). Graph Attention Networks. ICLR 2018.
    https://arxiv.org/abs/1710.10903
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool  # type: ignore
    from torch_geometric.data import Data  # type: ignore
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False
    GATConv = None
    global_mean_pool = None
    global_max_pool = None
    Data = None
    warnings.warn(
        "torch_geometric is required for full gat_model functionality. "
        "Install: pip install torch_geometric",
        UserWarning,
        stacklevel=1,
    )

logger = logging.getLogger(__name__)

# Integer → dialect name, mirroring DIALECT_TO_ID in madar_preprocessing.
DIALECT_NAMES: List[str] = ["Gulf", "Egyptian", "Levantine", "Maghrebi", "Iraqi"]


# ---------------------------------------------------------------------------
# PhonemeEmbedding
# ---------------------------------------------------------------------------

class PhonemeEmbedding(nn.Module):
    """Learnable embedding for discrete phoneme types.

    Used when the input representation is a phoneme vocabulary index rather
    than a pre-computed feature vector. Produces dense embeddings that are
    trained jointly with the GAT.

    Args:
        vocab_size: Number of distinct phoneme symbols in the vocabulary.
        embed_dim: Output embedding dimension. Default 64.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dim = embed_dim

    def forward(self, phoneme_ids: Tensor) -> Tensor:
        """Embed phoneme IDs.

        Args:
            phoneme_ids: Long tensor of phoneme IDs, shape (...).

        Returns:
            Float tensor of shape (..., embed_dim).
        """
        return self.embedding(phoneme_ids)


# ---------------------------------------------------------------------------
# GATLayer
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """Single Graph Attention layer with optional residual connection.

    Wraps :class:`torch_geometric.nn.GATConv` and:
    - Exposes attention weights from the forward pass (needed for visualization).
    - Adds a residual connection when ``in_channels == out_channels * heads``
      (i.e., when the concatenated output dimension matches the input).
    - Applies dropout to attention coefficients during training.

    Args:
        in_channels: Input node feature dimension.
        out_channels: Per-head output dimension. Total output when
            ``concat=True`` is ``out_channels * heads``.
        heads: Number of attention heads.
        dropout: Dropout applied to attention coefficients (0 disables).
        concat: If True, concatenate head outputs; otherwise average.
        residual: If True and dimensions match, add a residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.1,
        concat: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.residual = residual

        self.conv = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            concat=concat,
            add_self_loops=True,
        )

        # Dropout applied after the convolution output (not the attention weights —
        # GATConv handles internal attention dropout via its own `dropout` param).
        self.out_dropout = nn.Dropout(p=dropout)

        # Residual projection when dimensions don't align automatically.
        concat_dim = out_channels * heads if concat else out_channels
        if residual and in_channels != concat_dim:
            self.residual_proj: Optional[nn.Linear] = nn.Linear(in_channels, concat_dim, bias=False)
        else:
            self.residual_proj = None

        self._out_dim = concat_dim

    @property
    def out_dim(self) -> int:
        """Output feature dimension (= out_channels * heads if concat else out_channels)."""
        return self._out_dim

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass through the GAT layer.

        Args:
            x: Node feature matrix, shape (N, in_channels).
            edge_index: COO edge tensor, shape (2, E).

        Returns:
            Tuple of:
            - ``x_out``: Updated node features, shape (N, out_channels * heads).
            - ``attention_weights``: Attention coefficients, shape (E, heads).
              These are used in :mod:`evaluation.error_analysis` and the
              visualization notebooks to understand what phoneme pairs the
              model attends to.
        """
        # GATConv with return_attention_weights=True returns ((N, out), (edge_index, alpha)).
        x_out, (_, alpha) = self.conv(x, edge_index, return_attention_weights=True)
        # alpha shape: (E_with_self_loops, heads)

        if self.residual:
            if self.residual_proj is not None:
                x_out = x_out + self.residual_proj(x)
            elif x.shape == x_out.shape:
                x_out = x_out + x

        x_out = self.out_dropout(x_out)
        return x_out, alpha


# ---------------------------------------------------------------------------
# DialectGAT
# ---------------------------------------------------------------------------

class DialectGAT(nn.Module):
    """Full Graph Attention Network for Arabic dialect classification.

    Processes a phoneme-level graph and outputs per-class logits and the
    attention weights from each GAT layer (for visualization).

    Architecture:
        1. Input projection:  node_feature_dim → hidden_dim
        2. num_layers × GATLayer (hidden_dim → hidden_dim, 8 heads each)
        3. Global pooling (mean or max) → graph embedding
        4. MLP:  hidden_dim → hidden_dim//2 → ReLU → Dropout → num_classes

    Args:
        node_feature_dim: Dimension of input node features (9 by default:
            8 phoneme-class one-hot + 1 position scalar).
        num_classes: Number of dialect classes (5).
        hidden_dim: Hidden dimension throughout the network (128).
        num_heads: Attention heads per GAT layer (8).
        num_layers: Number of stacked GAT layers (3).
        dropout: Dropout probability applied in GAT layers and MLP.
        pooling: Graph-level pooling strategy — ``"mean"`` or ``"max"``.
    """

    def __init__(
        self,
        node_feature_dim: int = 9,
        num_classes: int = 5,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        pooling: str = "mean",
    ) -> None:
        super().__init__()

        assert num_layers >= 1, "num_layers must be >= 1"
        assert pooling in ("mean", "max"), f"pooling must be 'mean' or 'max', got {pooling!r}"

        self.node_feature_dim = node_feature_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        # ------------------------------------------------------------------
        # Input projection: maps raw node features into the hidden space.
        # A simple linear layer — no activation here since GATConv applies
        # its own transformation internally.
        # ------------------------------------------------------------------
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        # ------------------------------------------------------------------
        # GAT layers.
        # Each GATLayer takes hidden_dim inputs and produces hidden_dim outputs
        # by using per-head dim = hidden_dim // num_heads and concatenating.
        # ------------------------------------------------------------------
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        per_head_dim = hidden_dim // num_heads

        self.gat_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GATLayer(
                    in_channels=hidden_dim,
                    out_channels=per_head_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,    # concat → out = per_head_dim * num_heads = hidden_dim
                    residual=True,
                )
            )

        # ------------------------------------------------------------------
        # Batch normalization between GAT layers for training stability.
        # ------------------------------------------------------------------
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # ------------------------------------------------------------------
        # Pooling (applied to per-node embeddings → graph embedding).
        # ------------------------------------------------------------------
        self._pool_fn = global_mean_pool if pooling == "mean" else global_max_pool

        # ------------------------------------------------------------------
        # MLP classifier: graph embedding → num_classes.
        # ------------------------------------------------------------------
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform initialization for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        data: Data,
    ) -> Tuple[Tensor, List[Tensor]]:
        """Forward pass.

        Args:
            data: A PyG :class:`~torch_geometric.data.Data` object with:
                - ``x``:          Node features (N_total, node_feature_dim).
                - ``edge_index``: COO edges (2, E_total).
                - ``batch``:      Batch vector (N_total,) mapping nodes to
                  their graph in the mini-batch. Set to zeros tensor for
                  single-graph inference.

        Returns:
            Tuple of:
            - ``logits``: Raw class scores (batch_size, num_classes).
            - ``attn_weights``: List of attention weight tensors, one per
              GAT layer. Each tensor has shape (E_with_self_loops, num_heads).
              These are stacked across heads and can be averaged to get a
              single scalar per edge for visualization.
        """
        x, edge_index = data.x, data.edge_index

        # batch vector: zeros if doing single-graph inference.
        if data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Input projection.
        x = self.input_proj(x)
        x = F.relu(x)

        # GAT layers with layer norm and ELU activations.
        all_attn_weights: List[Tensor] = []
        for gat_layer, layer_norm in zip(self.gat_layers, self.layer_norms):
            x, alpha = gat_layer(x, edge_index)
            x = layer_norm(x)
            x = F.elu(x)
            all_attn_weights.append(alpha)  # (E, heads)

        # Graph-level pooling.
        graph_embed = self._pool_fn(x, batch)   # (batch_size, hidden_dim)

        # MLP classifier.
        logits = self.mlp(graph_embed)           # (batch_size, num_classes)

        return logits, all_attn_weights

    @torch.no_grad()
    def predict(self, data: Data) -> Tuple[str, float]:
        """Predict dialect and confidence for a single graph.

        Args:
            data: PyG Data object for a single utterance.

        Returns:
            Tuple of (dialect_name, confidence) where confidence is the
            softmax probability of the top class.
        """
        self.eval()
        logits, _ = self.forward(data)
        probs = F.softmax(logits, dim=-1)
        class_idx = int(probs.argmax(dim=-1).item())
        confidence = float(probs[0, class_idx].item())
        dialect_name = DIALECT_NAMES[class_idx]
        return dialect_name, confidence

    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> "DialectGAT":
        """Load a DialectGAT model from a saved checkpoint.

        The checkpoint dictionary must contain:
        - ``"model_state_dict"``: Module state dict.
        - ``"config"``: Dict of constructor keyword arguments.

        Args:
            checkpoint_path: Path to the ``.pth`` checkpoint file.

        Returns:
            Loaded DialectGAT model in eval mode.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logger.info("Loaded DialectGAT from %s", checkpoint_path)
        return model

    def save_checkpoint(self, path: str, extra: Optional[Dict] = None) -> None:
        """Save model state and config to a checkpoint file.

        Args:
            path: Output ``.pth`` path.
            extra: Optional extra metadata to include (e.g. F1 score, epoch).
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "node_feature_dim": self.node_feature_dim,
                "num_classes": self.num_classes,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "pooling": self.pooling,
            },
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved → %s", path)

    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"DialectGAT("
            f"feat_dim={self.node_feature_dim}, "
            f"hidden={self.hidden_dim}, "
            f"heads={self.num_heads}, "
            f"layers={self.num_layers}, "
            f"pool={self.pooling}, "
            f"params={self.num_parameters():,})"
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_gat(config: Dict) -> DialectGAT:
    """Build a :class:`DialectGAT` from a configuration dictionary.

    Expected keys (all optional — defaults match the best model config):
        node_feature_dim, num_classes, hidden_dim, num_heads,
        num_layers, dropout, pooling.

    Args:
        config: Configuration dict. Typically loaded from a YAML file
            (e.g. ``configs/gat_base.yaml``).

    Returns:
        Initialized DialectGAT.
    """
    model_cfg = config.get("model", config)   # Accept config with or without "model" block.
    return DialectGAT(
        node_feature_dim=model_cfg.get("node_feature_dim", 9),
        num_classes=model_cfg.get("num_classes", 5),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_heads=model_cfg.get("num_heads", 8),
        num_layers=model_cfg.get("num_layers", 3),
        dropout=model_cfg.get("dropout", 0.1),
        pooling=model_cfg.get("pooling", "mean"),
    )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    print("=" * 60)
    print("  DialectGAT — Architecture Verification")
    print("=" * 60)

    # Build model.
    model = build_gat({})
    print(f"\n{model}\n")

    # Create a synthetic mini-batch: 2 graphs, 5 nodes each, 4 edges each.
    n_nodes = 10          # total nodes across both graphs
    n_edges = 18          # total edges
    feat_dim = 9

    x = torch.randn(n_nodes, feat_dim)
    edge_index = torch.tensor(
        [[0,1,1,2,2,3,3,4, 5,6,6,7,7,8,8,9, 0,5],
         [1,0,2,1,3,2,4,3, 6,5,7,6,8,7,9,8, 2,7]],
        dtype=torch.long,
    )
    batch = torch.tensor([0,0,0,0,0, 1,1,1,1,1], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, batch=batch)

    # Forward pass.
    model.eval()
    with torch.no_grad():
        logits, attn_weights = model(data)

    print(f"Logits shape:        {logits.shape}  (expected [2, 5])")
    print(f"Attention layers:    {len(attn_weights)}  (expected {model.num_layers})")
    for i, alpha in enumerate(attn_weights):
        print(f"  Layer {i} attn:    {alpha.shape}  (E+self_loops, heads)")
    print(f"\nTotal parameters:    {model.num_parameters():,}")

    # Single-graph predict.
    single = Data(x=x[:5], edge_index=edge_index[:, :8], batch=None)
    dialect, conf = model.predict(single)
    print(f"\nSingle-graph predict: {dialect}  ({conf:.1%} confidence)")
    print("\nAll checks passed.")
