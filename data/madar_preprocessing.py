"""
data/madar_preprocessing.py

MADAR corpus preprocessing pipeline.

The MADAR corpus (Bouamor et al., 2018) provides parallel Arabic sentences
across 25 cities. This module:
  - Loads the TSV files
  - Maps 25 cities → 5 dialect groups
  - Cleans Arabic text (diacritics, alef normalization, etc.)
  - Performs stratified train/val/test splits
  - Reports per-dialect corpus statistics

Usage:
    python data/madar_preprocessing.py \
        --data_dir data/raw \
        --output_dir data/processed \
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants: dialect mapping
# ---------------------------------------------------------------------------

#: Maps dialect group name → list of city names as they appear in MADAR TSV.
DIALECTS: Dict[str, List[str]] = {
    "Gulf": [
        "Abu Dhabi", "Doha", "Dubai", "Kuwait", "Muscat", "Riyadh", "Sanaa"
    ],
    "Egyptian": ["Cairo", "Alexandria"],
    "Levantine": ["Beirut", "Damascus", "Amman", "Jerusalem"],
    "Maghrebi": ["Tunis", "Algiers", "Rabat", "Tripoli"],
    "Iraqi": ["Baghdad", "Mosul", "Basra"],
}

#: Dialect name → integer label.
DIALECT_TO_ID: Dict[str, int] = {
    "Gulf": 0,
    "Egyptian": 1,
    "Levantine": 2,
    "Maghrebi": 3,
    "Iraqi": 4,
}

#: Reverse mapping: integer → dialect name.
ID_TO_DIALECT: Dict[int, str] = {v: k for k, v in DIALECT_TO_ID.items()}

# Build city → dialect lookup for fast access.
_CITY_TO_DIALECT: Dict[str, str] = {}
for _dialect, _cities in DIALECTS.items():
    for _city in _cities:
        _CITY_TO_DIALECT[_city] = _dialect

# Unicode ranges for Arabic diacritics (harakat + other marks).
_DIACRITIC_RE = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]")
# Non-Arabic, non-space characters to strip.
_NON_ARABIC_RE = re.compile(r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]")
# Collapse multiple spaces.
_MULTI_SPACE_RE = re.compile(r"\s{2,}")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_arabic_text(
    text: str,
    normalize_ta_marbuta: bool = False,
) -> str:
    """Clean and normalize an Arabic text string.

    Steps applied, in order:
    1. Strip diacritics (harakat, U+064B–U+065F and extended marks).
    2. Normalize alef variants (أ إ آ → ا).
    3. Normalize dotless ya (ى → ي).
    4. Optionally normalize ta marbuta (ة → ه).
    5. Strip non-Arabic, non-space characters.
    6. Collapse extra whitespace.

    Args:
        text: Raw Arabic string.
        normalize_ta_marbuta: If True, map ة → ه. This is controversial —
            ta marbuta can carry morphological information — so it defaults
            to False. Set via CLI flag if desired.

    Returns:
        Cleaned Arabic string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Remove diacritics.
    text = _DIACRITIC_RE.sub("", text)

    # 2. Normalize alef variants.
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")

    # 3. Normalize ya.
    text = text.replace("ى", "ي")

    # 4. Optionally normalize ta marbuta.
    if normalize_ta_marbuta:
        text = text.replace("ة", "ه")

    # 5. Strip non-Arabic, non-space characters (punctuation, latin, numbers).
    text = _NON_ARABIC_RE.sub("", text)

    # 6. Collapse whitespace.
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_madar_corpus(
    data_dir: str | Path,
    split: str = "train",
    normalize_ta_marbuta: bool = False,
) -> pd.DataFrame:
    """Load a MADAR corpus TSV file and return a cleaned DataFrame.

    The function tries several common MADAR filename conventions:
      - MADAR-Corpus-26-{split}.tsv
      - madar_{split}.tsv
      - {split}.tsv

    Args:
        data_dir: Directory containing MADAR TSV files.
        split: One of ``"train"``, ``"dev"``, ``"test"``.
        normalize_ta_marbuta: Passed to :func:`clean_arabic_text`.

    Returns:
        DataFrame with columns: ``text``, ``city``, ``dialect``, ``dialect_id``.

    Raises:
        FileNotFoundError: If no MADAR TSV is found in *data_dir*.
    """
    data_dir = Path(data_dir)
    candidates = [
        data_dir / f"MADAR-Corpus-26-{split}.tsv",
        data_dir / f"MADAR-Corpus-26-{split.upper()}.tsv",
        data_dir / f"madar_{split}.tsv",
        data_dir / f"{split}.tsv",
    ]

    tsv_path: Optional[Path] = None
    for candidate in candidates:
        if candidate.exists():
            tsv_path = candidate
            break

    if tsv_path is None:
        raise FileNotFoundError(
            f"No MADAR TSV found in {data_dir!r} for split={split!r}. "
            f"Tried: {[str(c) for c in candidates]}. "
            f"See data/README.md for download instructions."
        )

    logger.info("Loading MADAR split=%s from %s", split, tsv_path)

    # MADAR TSV has no header; columns are sentence, city, (sometimes) dialect.
    raw_df = pd.read_csv(tsv_path, sep="\t", header=None, dtype=str)
    raw_df.columns = list(raw_df.columns)  # ensure RangeIndex

    if raw_df.shape[1] == 3:
        raw_df.columns = ["text", "city", "dialect_raw"]
    elif raw_df.shape[1] == 2:
        raw_df.columns = ["text", "city"]
    else:
        raise ValueError(
            f"Unexpected number of columns ({raw_df.shape[1]}) in {tsv_path}. "
            f"Expected 2 or 3."
        )

    # Drop rows with null text or unknown city.
    raw_df = raw_df.dropna(subset=["text", "city"])
    raw_df["city"] = raw_df["city"].str.strip()

    # Map city → dialect group.
    raw_df["dialect"] = raw_df["city"].map(_CITY_TO_DIALECT)
    unknown_cities = raw_df["dialect"].isna()
    if unknown_cities.any():
        unknown = raw_df.loc[unknown_cities, "city"].unique().tolist()
        logger.warning(
            "Skipping %d rows with unrecognized cities: %s",
            unknown_cities.sum(),
            unknown,
        )
        raw_df = raw_df[~unknown_cities].copy()

    # Clean text.
    raw_df["text"] = raw_df["text"].apply(
        lambda t: clean_arabic_text(t, normalize_ta_marbuta=normalize_ta_marbuta)
    )

    # Drop empty texts after cleaning.
    raw_df = raw_df[raw_df["text"].str.len() > 0].copy()

    # Dialect ID.
    raw_df["dialect_id"] = raw_df["dialect"].map(DIALECT_TO_ID)

    result = raw_df[["text", "city", "dialect", "dialect_id"]].reset_index(drop=True)

    logger.info(
        "Loaded %d sentences for split=%s. Class distribution:\n%s",
        len(result),
        split,
        result["dialect"].value_counts().to_string(),
    )
    return result


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def split_dataset(
    df: pd.DataFrame,
    train: float = 0.72,
    val: float = 0.14,
    test: float = 0.14,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train / val / test using stratified sampling.

    Args:
        df: Full DataFrame (from :func:`load_madar_corpus` or concatenated).
        train: Fraction for training set.
        val: Fraction for validation set.
        test: Fraction for test set.
        seed: Random seed for reproducibility.
        stratify: If True, maintain dialect class proportions in each split.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If fractions do not sum to 1.0 (within 1e-6 tolerance).
    """
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total:.6f}.")

    stratify_col = df["dialect"] if stratify else None

    train_df, temp_df = train_test_split(
        df,
        test_size=(val + test),
        random_state=seed,
        stratify=stratify_col,
    )

    # Now split temp into val and test.
    relative_val = val / (val + test)
    stratify_temp = temp_df["dialect"] if stratify else None

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - relative_val),
        random_state=seed,
        stratify=stratify_temp,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = split_df["dialect"].value_counts()
        logger.info("  %s distribution:\n%s", split_name, dist.to_string())

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_dialect_statistics(df: pd.DataFrame) -> dict:
    """Compute per-dialect and overall corpus statistics.

    Args:
        df: DataFrame with ``text``, ``dialect``, ``dialect_id`` columns.

    Returns:
        Dictionary with ``"per_dialect"`` and ``"overall"`` keys.
    """
    per_dialect: Dict[str, dict] = {}
    for dialect in DIALECT_TO_ID:
        subset = df[df["dialect"] == dialect]
        if len(subset) == 0:
            per_dialect[dialect] = {"count": 0}
            continue
        token_counts = subset["text"].str.split().str.len()
        word_sets = subset["text"].str.split().apply(set)
        unique_vocab: set = set().union(*word_sets) if len(word_sets) > 0 else set()
        per_dialect[dialect] = {
            "count": len(subset),
            "mean_length_words": round(float(token_counts.mean()), 2),
            "std_length_words": round(float(token_counts.std()), 2),
            "unique_words": len(unique_vocab),
        }

    overall_counts = pd.Series(
        {d: per_dialect[d].get("count", 0) for d in DIALECT_TO_ID},
        name="count",
    )
    max_count = overall_counts.max()
    min_count = overall_counts[overall_counts > 0].min() if (overall_counts > 0).any() else 1

    stats = {
        "per_dialect": per_dialect,
        "overall": {
            "total": len(df),
            "num_dialects": len(DIALECT_TO_ID),
            "class_balance_ratio": round(float(min_count / max_count), 4),
        },
    }

    # Print formatted report.
    print("\n" + "=" * 60)
    print("  CORPUS STATISTICS")
    print("=" * 60)
    print(f"  Total sentences : {stats['overall']['total']:,}")
    print(f"  Dialects        : {stats['overall']['num_dialects']}")
    print(f"  Balance ratio   : {stats['overall']['class_balance_ratio']:.4f}  (min/max class size)")
    print("-" * 60)
    print(f"  {'Dialect':<12} {'Count':>8} {'Mean len':>10} {'Unique vocab':>14}")
    print("-" * 60)
    for dialect, info in per_dialect.items():
        if info.get("count", 0) == 0:
            print(f"  {dialect:<12} {'0':>8} {'—':>10} {'—':>14}")
        else:
            print(
                f"  {dialect:<12} {info['count']:>8,} "
                f"{info['mean_length_words']:>10.1f} "
                f"{info['unique_words']:>14,}"
            )
    print("=" * 60 + "\n")

    return stats


# ---------------------------------------------------------------------------
# Save processed data
# ---------------------------------------------------------------------------

def save_processed(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Save train/val/test DataFrames as CSVs to *output_dir*.

    Args:
        train_df: Training split.
        val_df: Validation split.
        test_df: Test split.
        output_dir: Directory to write CSV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = ["text", "city", "dialect", "dialect_id"]
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = output_dir / f"{name}.csv"
        split_df[columns].to_csv(out_path, index=False, encoding="utf-8")
        logger.info("Saved %s → %s", name, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess MADAR corpus into 5-dialect train/val/test splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Directory containing raw MADAR TSV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to write processed CSV files.",
    )
    parser.add_argument(
        "--normalize_ta_marbuta",
        action="store_true",
        default=False,
        help="Map ة → ه during text cleaning (off by default).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified split.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.72,
        help="Fraction of data for training.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.14,
        help="Fraction of data for validation.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load all splits if they exist separately; otherwise load what's available
    # and split programmatically.
    dfs: List[pd.DataFrame] = []
    for split_name in ["train", "dev", "test"]:
        try:
            df = load_madar_corpus(
                args.data_dir,
                split=split_name,
                normalize_ta_marbuta=args.normalize_ta_marbuta,
            )
            dfs.append(df)
        except FileNotFoundError:
            logger.info("Split '%s' not found, skipping.", split_name)

    if not dfs:
        raise SystemExit(
            "No MADAR files found. See data/README.md for download instructions."
        )

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["text", "dialect"]).reset_index(drop=True)

    logger.info("Combined corpus: %d sentences after dedup.", len(full_df))

    # Show corpus statistics.
    compute_dialect_statistics(full_df)

    # Split.
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    train_df, val_df, test_df = split_dataset(
        full_df,
        train=args.train_ratio,
        val=args.val_ratio,
        test=test_ratio,
        seed=args.seed,
        stratify=True,
    )

    # Save.
    save_processed(train_df, val_df, test_df, args.output_dir)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
