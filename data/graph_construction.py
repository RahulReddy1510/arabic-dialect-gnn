"""
data/graph_construction.py

Converts phoneme sequences (from camel_pipeline.py) into
PyTorch Geometric Data objects for training.

Separated cleanly from CAMeL Tools — works with whatever
(phonemes, edges, features) tuple the pipeline produces.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data

if TYPE_CHECKING:
    import pandas as pd
    from data.camel_pipeline import CAMeLProcessor, text_to_phoneme_graph

logger = logging.getLogger(__name__)


try:
    import networkx as nx  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib.patches as mpatches  # type: ignore
    _VIZ_AVAILABLE = True
except ImportError:
    _VIZ_AVAILABLE = False

# ---------------------------------------------------------------------------
# Phoneme class metadata for visualization
# ---------------------------------------------------------------------------

PHONEME_CLASSES = ["stop", "fricative", "affricate", "nasal", "liquid", "glide", "vowel", "pharyngeal"]

_CLASS_COLORS = {
    "stop": "#4C72B0",
    "fricative": "#DD8452",
    "affricate": "#55A868",
    "nasal": "#C44E52",
    "liquid": "#8172B2",
    "glide": "#937860",
    "vowel": "#64B5CD",
    "pharyngeal": "#DA8BC3",
}
_CLASS_COLOR_LIST = list(_CLASS_COLORS.values())


def _class_index_from_feature(feature_vec: List[float]) -> int:
    if not feature_vec:
        return 0
    return int(max(range(len(feature_vec)), key=lambda i: feature_vec[i]))


# ---------------------------------------------------------------------------
# Core: phonemes → PyG Data
# ---------------------------------------------------------------------------

def phonemes_to_pyg_graph(
    phonemes: List[str],
    edges: List[Tuple[int, int]],
    features: List[List[float]],
    label: int,
) -> Data:
    """Build a PyTorch Geometric Data object from a phoneme graph.

    Args:
        phonemes: Ordered list of IPA-like phoneme strings (length N).
        edges: List of (src, dst) int tuples (bidirectional preferred).
        features: List of N feature vectors (each length 8).
        label: Integer dialect class label (0–4).

    Returns:
        :class:`torch_geometric.data.Data` with attributes:

        - ``x``: Float32 node feature matrix (N × feature_dim).
        - ``edge_index``: Long COO edge tensor (2 × E).
        - ``y``: Scalar long label tensor.
        - ``num_nodes``: N.
        - ``phonemes``: List of phoneme strings (for visualization).
    """
    from torch_geometric.data import Data  # type: ignore

    n_nodes = len(phonemes)

    if features:
        x = torch.tensor(features, dtype=torch.float32)
    else:
        x = torch.ones((n_nodes, 8), dtype=torch.float32) / 8.0

    if edges:
        src, dst = zip(*edges)
        edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    y = torch.tensor(label, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=n_nodes)
    data.phonemes = phonemes
    return data


# ---------------------------------------------------------------------------
# Positional features
# ---------------------------------------------------------------------------

def add_positional_features(
    graph_data: Data,
    max_position: int = 100,
) -> Data:
    """Append a normalized scalar position to each node's feature vector.

    Phoneme position in the utterance carries information about stressed
    syllables and word-boundary placement.

    Args:
        graph_data: PyG Data object with ``x`` set.
        max_position: Normalisation ceiling — positions are divided by
            (max_position - 1) and clipped to [0, 1].

    Returns:
        Same Data object with ``x`` extended by one column.
    """
    n = graph_data.num_nodes
    if n == 0:
        return graph_data
    positions = torch.arange(n, dtype=torch.float32)
    positions = (positions / max(max_position - 1, 1)).clamp(0.0, 1.0).unsqueeze(1)
    graph_data.x = torch.cat([graph_data.x, positions], dim=1)
    return graph_data


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_graph(graph_data: Data) -> Data:
    """Z-score normalize node features per feature dimension.

    Zero-variance dimensions are left unchanged (std clipped to 1e-8).

    Args:
        graph_data: PyG Data object.

    Returns:
        Same Data object with normalized ``x``.
    """
    x = graph_data.x
    if x is None or x.numel() == 0:
        return graph_data
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp(min=1e-8)
    graph_data.x = (x - mean) / std
    return graph_data


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------

def batch_construct_graphs(
    df: pd.DataFrame,
    processor: CAMeLProcessor,
    output_dir: str | Path,
    split: str = "train",
) -> List[Path]:
    """Construct and save graphs for every row in *df*.

    Args:
        df: DataFrame with ``text`` and ``dialect_id`` columns.
        processor: :class:`data.camel_pipeline.CAMeLProcessor` instance.
        output_dir: Root directory. Files saved under ``{output_dir}/{split}/``.
        split: Split name used as subdirectory.

    Returns:
        List of saved ``.pt`` file paths.
    """
    from tqdm import tqdm

    out_dir = Path(output_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    failures = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Graphs/{split}"):
        try:
            phonemes, edges, features = text_to_phoneme_graph(str(row["text"]), processor)
            if not phonemes:
                failures += 1
                continue
            graph = phonemes_to_pyg_graph(phonemes, edges, features, label=int(row["dialect_id"]))
            graph = add_positional_features(graph)
            graph = normalize_graph(graph)
            out_path = out_dir / f"{idx:06d}.pt"
            torch.save(graph, out_path)
            saved_paths.append(out_path)
        except Exception as exc:
            logger.warning("Failed idx=%d: %s", idx, exc)
            failures += 1

    logger.info("[%s] Saved %d graphs, %d failures.", split, len(saved_paths), failures)
    return saved_paths


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_graph_dataset(data_dir: str | Path, split: str) -> List[Data]:
    """Load all ``.pt`` graph files for a given split.

    Args:
        data_dir: Root data directory (contains ``{split}/`` subdir).
        split: One of ``"train"``, ``"val"``, ``"test"``.

    Returns:
        Sorted list of PyG Data objects.
    """
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Graph directory not found: {split_dir}. "
            "Run graph_construction.py --processed_dir ... --output_dir ... first."
        )
    pt_files = sorted(split_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {split_dir}.")
    dataset = []
    for f in pt_files:
        try:
            dataset.append(torch.load(f, weights_only=False))
        except Exception as exc:
            logger.warning("Could not load %s: %s", f, exc)
    logger.info("Loaded %d graphs from %s/%s", len(dataset), data_dir, split)
    return dataset


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_phoneme_graph(
    graph_data: Data,
    phoneme_list: List[str],
    title: str = "Phoneme Graph",
    save_path: Optional[str | Path] = None,
    attention_weights: Optional[Tensor] = None,
) -> None:
    """Render a phoneme graph using networkx + matplotlib.

    Nodes are colored by phoneme class. Edge width is proportional to
    attention weight if provided. If visualization libraries are not
    installed, logs a warning and returns early.

    Args:
        graph_data: PyG Data object.
        phoneme_list: Phoneme strings for node labels.
        title: Plot title.
        save_path: Save figure here instead of showing if provided.
        attention_weights: Optional (E,) tensor of edge attention scores.
    """
    if not _VIZ_AVAILABLE:
        logger.warning("networkx / matplotlib not installed; skipping visualization.")
        return

    n_nodes = graph_data.num_nodes
    if n_nodes == 0:
        logger.warning("Empty graph — nothing to visualize.")
        return

    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))

    node_colors = []
    for i in range(n_nodes):
        feat = graph_data.x[i].tolist() if graph_data.x is not None else []
        class_idx = _class_index_from_feature(feat)
        node_colors.append(_CLASS_COLOR_LIST[class_idx % len(_CLASS_COLOR_LIST)])

    edge_index = graph_data.edge_index
    edge_list = []
    if edge_index is not None and edge_index.shape[1] > 0:
        for e in range(edge_index.shape[1]):
            edge_list.append((int(edge_index[0, e]), int(edge_index[1, e])))
            G.add_edge(int(edge_index[0, e]), int(edge_index[1, e]))

    if attention_weights is not None and len(attention_weights) == len(edge_list):
        attn = attention_weights.detach().cpu().numpy()
        attn_n = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        edge_widths = [0.5 + 4.0 * float(w) for w in attn_n]
    else:
        edge_widths = [1.0] * len(edge_list)

    labels = {i: phoneme_list[i] if i < len(phoneme_list) else str(i) for i in range(n_nodes)}
    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(max(8, n_nodes * 0.8), 5))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edgelist=edge_list, width=edge_widths,
        edge_color="#888888", arrows=True, ax=ax,
    )
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax)
    patches = [
        mpatches.Patch(color=_CLASS_COLORS[c], label=c) for c in PHONEME_CLASSES
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.8)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Saved visualization → %s", save_path)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse
    import pandas as pd
    from data.camel_pipeline import CAMeLProcessor

    parser = argparse.ArgumentParser(
        description="Build phoneme graphs from processed MADAR CSV files."
    )
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--output_dir", default="data/graphs")
    args = parser.parse_args()

    processor = CAMeLProcessor()
    for split in ["train", "val", "test"]:
        csv_path = Path(args.processed_dir) / f"{split}.csv"
        if not csv_path.exists():
            logger.warning("No CSV for split=%s at %s, skipping.", split, csv_path)
            continue
        import pandas as pd  # noqa: F811
        df = pd.read_csv(csv_path)
        logger.info("Building graphs for split=%s (%d rows).", split, len(df))
        batch_construct_graphs(df, processor, args.output_dir, split=split)


if __name__ == "__main__":
    _main()
