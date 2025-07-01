"""
models/graph_utils.py

Graph-level helper functions: statistics, attention analysis, and
side-by-side text/phoneme/graph comparison utilities.

These are the kind of functions that accumulate organically over a research
project — I kept adding to this file whenever I needed a quick diagnostic
that didn't belong in a notebook but also didn't belong in the model itself.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt  # type: ignore
    _MPL = True
except ImportError:
    _MPL = False

DIALECT_NAMES: List[str] = ["Gulf", "Egyptian", "Levantine", "Maghrebi", "Iraqi"]


# ---------------------------------------------------------------------------
# Graph statistics
# ---------------------------------------------------------------------------

def compute_graph_stats(dataset) -> Dict:
    """Compute mean/std graph-level statistics for a dataset.

    Args:
        dataset: Any iterable of PyG Data objects (ArabicDialectGraphDataset
            or a plain list from load_graph_dataset).

    Returns:
        Dictionary with:
        - ``"overall"``: mean/std of num_nodes, num_edges, density.
        - ``"per_dialect"``: same statistics broken down by class label.
    """
    num_nodes_list: List[int] = []
    num_edges_list: List[int] = []
    densities: List[float] = []
    per_dialect_nodes: Dict[str, List[int]] = {d: [] for d in DIALECT_NAMES}

    for data in dataset:
        n = int(data.num_nodes)
        e = int(data.edge_index.shape[1]) if data.edge_index is not None else 0
        # Graph density = |E| / (|V| * (|V| - 1)). Directed, so max edges = N*(N-1).
        max_edges = n * (n - 1) if n > 1 else 1
        density = e / max_edges

        num_nodes_list.append(n)
        num_edges_list.append(e)
        densities.append(density)

        label = int(data.y.item())
        if 0 <= label < len(DIALECT_NAMES):
            per_dialect_nodes[DIALECT_NAMES[label]].append(n)

    def _stats(arr: List[float]) -> Dict[str, float]:
        a = np.array(arr, dtype=float)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }

    overall = {
        "num_graphs": len(num_nodes_list),
        "num_nodes": _stats(num_nodes_list),
        "num_edges": _stats(num_edges_list),
        "density": _stats(densities),
    }

    per_dialect = {
        dialect: _stats(nodes) if nodes else {}
        for dialect, nodes in per_dialect_nodes.items()
    }

    stats = {"overall": overall, "per_dialect": per_dialect}
    _print_graph_stats(stats)
    return stats


def _print_graph_stats(stats: Dict) -> None:
    overall = stats["overall"]
    print("\n" + "=" * 60)
    print("  GRAPH DATASET STATISTICS")
    print("=" * 60)
    print(f"  Graphs          : {overall['num_graphs']:,}")
    print(f"  Nodes  mean±std : {overall['num_nodes']['mean']:.1f} ± {overall['num_nodes']['std']:.1f}")
    print(f"  Edges  mean±std : {overall['num_edges']['mean']:.1f} ± {overall['num_edges']['std']:.1f}")
    print(f"  Density mean    : {overall['density']['mean']:.4f}")
    print("-" * 60)
    print(f"  {'Dialect':<12} {'N graphs':>10} {'Mean nodes':>12}")
    print("-" * 60)
    for dialect, d_stats in stats["per_dialect"].items():
        if d_stats:
            print(f"  {dialect:<12} {d_stats.get('mean', 0):.1f}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Attention analysis helpers
# ---------------------------------------------------------------------------

def get_attention_subgraph(
    graph_data: "torch_geometric.data.Data",
    attention_weights: Tensor,
    top_k: int = 5,
) -> Tuple[Tensor, Tensor]:
    """Return the top-k most strongly attended edges.

    Args:
        graph_data: PyG Data object with ``edge_index``.
        attention_weights: Tensor of shape (E, num_heads). Typically the
            output of a single GATLayer (or averaged across layers).
        top_k: Number of top edges to return.

    Returns:
        Tuple of:
        - ``top_edge_index``: (2, top_k) COO tensor of top-k edges.
        - ``top_scores``: (top_k,) mean attention scores for those edges.
    """
    # Average attention across heads → scalar per edge.
    if attention_weights.dim() == 2:
        mean_attn = attention_weights.mean(dim=1)   # (E,)
    else:
        mean_attn = attention_weights

    k = min(top_k, mean_attn.shape[0])
    top_indices = torch.topk(mean_attn, k=k).indices   # (k,)

    top_edge_index = graph_data.edge_index[:, top_indices]   # (2, k)
    top_scores = mean_attn[top_indices]                      # (k,)

    return top_edge_index, top_scores


def phoneme_importance_analysis(
    model: "DialectGAT",
    dataset,
    dialect: str,
    max_samples: int = 100,
) -> List[Tuple[str, float]]:
    """Find which phonemes the model most attends to for a given dialect.

    Runs forward passes on a subset of the dataset filtered by *dialect*,
    collects layer-3 attention weights, and aggregates per-phoneme scores.

    Args:
        model: Trained :class:`models.gat_model.DialectGAT`.
        dataset: Iterable of PyG Data objects with ``phonemes`` attribute.
        dialect: Dialect name (one of DIALECT_NAMES).
        max_samples: Maximum number of samples from this dialect to process.

    Returns:
        List of (phoneme_string, mean_attention_score) sorted descending,
        representing the phonemes the final GAT layer attended to most.
    """
    if dialect not in DIALECT_NAMES:
        raise ValueError(f"Unknown dialect {dialect!r}. Choose from {DIALECT_NAMES}.")

    target_label = DIALECT_NAMES.index(dialect)
    model.eval()
    device = next(model.parameters()).device

    phoneme_scores: Dict[str, List[float]] = {}
    processed = 0

    with torch.no_grad():
        for data in dataset:
            if int(data.y.item()) != target_label:
                continue
            if processed >= max_samples:
                break

            data = data.to(device)
            _, attn_weights = model(data)
            # Use the last layer's attention.
            last_attn = attn_weights[-1].mean(dim=1).cpu()   # (E,)

            phonemes = getattr(data, "phonemes", [])
            edge_index = data.edge_index.cpu()

            for edge_idx in range(edge_index.shape[1]):
                src = int(edge_index[0, edge_idx])
                if src < len(phonemes):
                    phoneme = phonemes[src]
                    score = float(last_attn[edge_idx])
                    phoneme_scores.setdefault(phoneme, []).append(score)

            processed += 1

    ranked = sorted(
        [(ph, float(np.mean(scores))) for ph, scores in phoneme_scores.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def compare_graph_representations(
    text: str,
    processor: "CAMeLProcessor",
) -> None:
    """Print a side-by-side breakdown of text → tokens → phonemes → graph stats.

    Useful for debugging the pipeline and for the notebook demos.

    Args:
        text: Arabic text string.
        processor: A :class:`data.camel_pipeline.CAMeLProcessor` instance.
    """
    from data.camel_pipeline import text_to_phoneme_graph  # type: ignore

    tokens = processor.tokenize(text)
    phonemes, edges, features = text_to_phoneme_graph(text, processor)

    n_nodes = len(phonemes)
    n_edges = len(edges)
    max_edges = n_nodes * (n_nodes - 1) if n_nodes > 1 else 1
    density = n_edges / max_edges

    print("\n" + "─" * 60)
    print(f"  Text:      {text}")
    print(f"  Tokens:    {tokens}")
    print(f"  Phonemes:  {phonemes}")
    print(f"  Nodes:     {n_nodes}")
    print(f"  Edges:     {n_edges}  (density: {density:.3f})")
    print("─" * 60 + "\n")


# ---------------------------------------------------------------------------
# Histogram of graph sizes
# ---------------------------------------------------------------------------

def plot_graph_size_histogram(
    dataset,
    save_path: Optional[str] = None,
    title: str = "Graph Size Distribution",
) -> None:
    """Plot a histogram of node counts across the dataset.

    Args:
        dataset: Iterable of PyG Data objects.
        save_path: Save figure here if provided; otherwise show.
        title: Matplotlib figure title.
    """
    if not _MPL:
        logger.warning("matplotlib not installed; skipping histogram.")
        return

    sizes = [int(data.num_nodes) for data in dataset]
    labels = [int(data.y.item()) for data in dataset]

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    for label_idx, dialect in enumerate(DIALECT_NAMES):
        dialect_sizes = [s for s, l in zip(sizes, labels) if l == label_idx]
        ax.hist(
            dialect_sizes,
            bins=30,
            alpha=0.6,
            label=dialect,
            color=colors[label_idx],
        )

    ax.set_xlabel("Number of phoneme nodes per graph")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved histogram → %s", save_path)
    else:
        plt.show()
    plt.close(fig)
