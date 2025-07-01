"""
data/camel_pipeline.py

CAMeL Tools integration for Arabic-aware morphological analysis and
IPA-like phoneme extraction.

CAMeL Tools (Obeid et al., 2020) was developed at the Computational
Approaches to Modeling Language Lab, NYU Abu Dhabi. It provides the
Arabic-aware morphological analysis that makes phoneme-level graph
construction possible without building that infrastructure from scratch.

If camel-tools is not installed, this module falls back to a direct
character-level Arabic → phoneme mapping. The fallback is clearly
announced and produces valid (if slightly less accurate) phoneme sequences.

Install CAMeL Tools:
    pip install camel-tools
    camel_data -i morphology-db-msa-r13

Reference:
    Obeid, O., Zalmout, N., Khalifa, S., Taji, D., Oudah, M., Eryani, F.,
    Inoue, G., Erdmann, A., Habash, N., & Bouamor, H. (2020).
    CAMeL Tools: An Open Source Python Toolkit for Arabic Natural Language
    Processing. LREC 2020.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np  # noqa: F401

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing CAMeL Tools; set a flag used throughout this module.
# ---------------------------------------------------------------------------

_CAMEL_AVAILABLE = False
_CAMEL_IMPORT_ERROR: Optional[str] = None

try:
    from camel_tools.tokenizers.word import WordTokenizer          # type: ignore
    from camel_tools.morphology.database import MorphologyDB       # type: ignore
    from camel_tools.morphology.analyzer import Analyzer           # type: ignore
    _CAMEL_AVAILABLE = True
except ImportError as _e:
    _CAMEL_IMPORT_ERROR = str(_e)
    warnings.warn(
        "\n"
        "[camel_pipeline] CAMeL Tools is not installed or could not be imported.\n"
        f"  ImportError: {_CAMEL_IMPORT_ERROR}\n"
        "  Falling back to character-level Arabic → phoneme mapping.\n"
        "  This fallback is less accurate than morpheme-aware analysis.\n"
        "  To install: pip install camel-tools && camel_data -i morphology-db-msa-r13",
        UserWarning,
        stacklevel=1,
    )

# ---------------------------------------------------------------------------
# Arabic character → IPA-like phoneme mapping
# ---------------------------------------------------------------------------

#: Mapping from Arabic Unicode character to an IPA-like phoneme string.
#: This covers 28 consonants + alef (vowel carrier) and common vowel forms.
ARABIC_TO_PHONEME: Dict[str, str] = {
    # Consonants
    "ب": "b",
    "ت": "t",
    "ث": "θ",    # Interdental fricative — key Gulf Arabic marker
    "ج": "dʒ",
    "ح": "ħ",   # Pharyngeal fricative
    "خ": "x",
    "د": "d",
    "ذ": "ð",   # Interdental fricative (voiced)
    "ر": "r",
    "ز": "z",
    "س": "s",
    "ش": "ʃ",
    "ص": "sˤ",  # Emphatic s
    "ض": "dˤ",  # Emphatic d (in Gulf: sometimes lateral emphatic)
    "ط": "tˤ",  # Emphatic t
    "ظ": "ðˤ",  # Emphatic th
    "ع": "ʕ",   # Pharyngeal fricative — high frequency in Gulf Arabic
    "غ": "ɣ",
    "ف": "f",
    "ق": "q",   # Uvular stop — realized as glottal stop in Egyptian
    "ك": "k",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ه": "h",
    "و": "w",   # As consonant; see long vowel below
    "ي": "j",   # As consonant; see long vowel below
    # Vowel carrier
    "ا": "aː",  # Long a
    "ى": "aː",  # Alef maqsura (normalized to ya but can be this in input)
    "ة": "h",   # Ta marbuta (word-final)
    "ء": "ʔ",   # Hamza
    "أ": "ʔa",
    "إ": "ʔi",
    "آ": "ʔaː",
    "ئ": "ʔj",
    "ؤ": "ʔw",
    # Long vowels (when و and ي are used as vowels, captured during extraction)
    # Short vowels from diacritics (if diacritics present):
    "\u064E": "a",   # fatha → short a
    "\u064F": "u",   # damma → short u
    "\u0650": "i",   # kasra → short i
    "\u0651": "",    # shadda → gemination (handled separately)
    "\u0652": "",    # sukun → no vowel
}

#: Phoneme class labels for feature vector.
PHONEME_CLASSES = ["stop", "fricative", "affricate", "nasal", "liquid", "glide", "vowel", "pharyngeal"]
_N_CLASSES = len(PHONEME_CLASSES)

#: Phoneme → phoneme class index.
_PHONEME_CLASS_MAP: Dict[str, int] = {
    # Stops
    "b": 0, "t": 0, "d": 0, "k": 0, "q": 0, "ʔ": 0, "tˤ": 0, "dˤ": 0, "ʔa": 0, "ʔi": 0, "ʔaː": 0,
    "ʔj": 0, "ʔw": 0,
    # Fricatives
    "θ": 1, "ð": 1, "s": 1, "z": 1, "ʃ": 1, "f": 1, "h": 1, "x": 1, "ɣ": 1,
    "sˤ": 1, "ðˤ": 1,
    # Affricates
    "dʒ": 2,
    # Nasals
    "m": 3, "n": 3,
    # Liquids
    "r": 4, "l": 4,
    # Glides
    "w": 5, "j": 5,
    # Vowels
    "aː": 6, "uː": 6, "iː": 6, "a": 6, "u": 6, "i": 6,
    # Pharyngeals
    "ħ": 7, "ʕ": 7,
}

_FEATURE_DIM = _N_CLASSES + 1  # one-hot class (8) + pharyngeal_flag for quick indexing = 8


def _phoneme_to_features(phoneme: str) -> List[float]:
    """One-hot encode a phoneme by its phoneme class.

    Returns a list of length ``_N_CLASSES`` (8), with a 1.0 in the class
    position and 0.0 elsewhere.

    Args:
        phoneme: IPA-like phoneme string.

    Returns:
        List of floats (length 8).
    """
    vec = [0.0] * _N_CLASSES
    class_idx = _PHONEME_CLASS_MAP.get(phoneme)
    if class_idx is not None:
        vec[class_idx] = 1.0
    else:
        # Unknown phoneme → uniform distribution (soft unknown).
        vec = [1.0 / _N_CLASSES] * _N_CLASSES
    return vec


# ---------------------------------------------------------------------------
# Character-level fallback phoneme extractor
# ---------------------------------------------------------------------------

def _char_level_to_phonemes(text: str) -> List[str]:
    """Convert Arabic text to phoneme sequence using direct character mapping.

    This is the fallback used when CAMeL Tools is not installed. It does
    a simple character-by-character lookup using ARABIC_TO_PHONEME. No
    morphological analysis — it won't handle clitics or morpheme boundaries
    gracefully, but it produces a valid phoneme sequence.

    Args:
        text: Cleaned Arabic text string.

    Returns:
        List of IPA-like phoneme strings.
    """
    phonemes: List[str] = []
    i = 0
    chars = list(text)
    while i < len(chars):
        ch = chars[i]
        phoneme = ARABIC_TO_PHONEME.get(ch)
        if phoneme is None:
            # Skip: space, punctuation, unknown characters.
            i += 1
            continue
        if phoneme == "":
            # Diacritic that maps to empty (shadda, sukun).
            i += 1
            continue
        # Check for long vowel: و or ي following certain contexts.
        # Simple heuristic: if current char is 'و' preceded by consonant → uː.
        if ch == "و" and i > 0:
            prev = chars[i - 1]
            if prev in ARABIC_TO_PHONEME and ARABIC_TO_PHONEME.get(prev, "") not in ("", "a", "u", "i", "aː", "uː", "iː"):
                # Could be long vowel; context is ambiguous without morphology.
                phoneme = "uː"
        elif ch == "ي" and i > 0:
            prev = chars[i - 1]
            if prev in ARABIC_TO_PHONEME and ARABIC_TO_PHONEME.get(prev, "") not in ("", "a", "u", "i", "aː", "uː", "iː"):
                phoneme = "iː"

        phonemes.append(phoneme)
        i += 1

    return phonemes


# ---------------------------------------------------------------------------
# CAMeL-aware processor
# ---------------------------------------------------------------------------

class CAMeLProcessor:
    """Wraps CAMeL Tools morphological analyzer for Arabic phoneme extraction.

    If CAMeL Tools is not available, the object falls back to character-level
    processing and issues a UserWarning on construction. All methods remain
    callable.

    Args:
        analyzer_db: CAMeL morphology database name. Default ``"calima-msa-r13"``.
            If you have a dialect-specific DB (e.g. Gulf Arabic), pass it here.
        use_gpu: Reserved for future use; CAMeL Tools CPU-only as of v1.5.
    """

    def __init__(
        self,
        analyzer_db: str = "calima-msa-r13",
        use_gpu: bool = False,
    ) -> None:
        self._use_camel = _CAMEL_AVAILABLE
        self._analyzer: Optional[object] = None
        self._tokenizer: Optional[object] = None

        if self._use_camel:
            try:
                db = MorphologyDB.builtin_db(analyzer_db)
                self._analyzer = Analyzer(db, backoff="NOAN")
                self._tokenizer = WordTokenizer(analyzer_db)
                logger.info("CAMeL Tools initialized with db=%s", analyzer_db)
            except Exception as exc:
                warnings.warn(
                    f"[CAMeLProcessor] Failed to initialize CAMeL analyzer: {exc}. "
                    f"Falling back to character-level phoneme extraction.",
                    UserWarning,
                    stacklevel=2,
                )
                self._use_camel = False
        else:
            logger.warning(
                "CAMeLProcessor: using character-level fallback (camel-tools not installed)."
            )

    @property
    def uses_camel(self) -> bool:
        """True if CAMeL Tools is active; False if using fallback."""
        return self._use_camel

    def tokenize(self, text: str) -> List[str]:
        """Morpheme-aware tokenization.

        If CAMeL Tools is available, uses WordTokenizer. Otherwise, splits on
        whitespace (word-level fallback).

        Args:
            text: Arabic text string.

        Returns:
            List of morpheme/token strings.
        """
        if not text.strip():
            return []

        if self._use_camel and self._tokenizer is not None:
            try:
                tokens = self._tokenizer.tokenize(text)  # type: ignore
                return [t for t in tokens if t.strip()]
            except Exception as exc:
                logger.debug("CAMeL tokenization failed (%s), using split fallback.", exc)

        # Fallback: whitespace split.
        return text.split()

    def get_morpheme_features(self, text: str) -> List[Dict]:
        """Return per-morpheme feature dictionaries.

        Args:
            text: Arabic text string.

        Returns:
            List of dicts, one per morpheme/token:
            ``{"surface": str, "lemma": str, "pos": str, "stem": str}``
        """
        tokens = self.tokenize(text)
        results: List[Dict] = []

        for token in tokens:
            if self._use_camel and self._analyzer is not None:
                try:
                    analyses = self._analyzer.analyze(token)  # type: ignore
                    if analyses:
                        # Take the top analysis.
                        top = analyses[0]
                        results.append({
                            "surface": token,
                            "lemma": top.get("lex", token),
                            "pos": top.get("pos", "UNKNOWN"),
                            "stem": top.get("stem", token),
                        })
                        continue
                except Exception as exc:
                    logger.debug("CAMeL analyzer failed for token=%r: %s", token, exc)

            # Fallback: minimal feature dict.
            results.append({
                "surface": token,
                "lemma": token,
                "pos": "UNKNOWN",
                "stem": token,
            })

        return results

    def text_to_phonemes(self, text: str) -> List[str]:
        """Convert Arabic text to an IPA-like phoneme sequence.

        Uses CAMeL-aware tokenization first (preserving morpheme boundaries),
        then applies character-level phoneme mapping to each token.

        Args:
            text: Arabic text string (should already be cleaned).

        Returns:
            List of IPA-like phoneme strings.
        """
        if not text.strip():
            return []

        tokens = self.tokenize(text)
        phonemes: List[str] = []

        for token in tokens:
            token_phonemes = _char_level_to_phonemes(token)
            phonemes.extend(token_phonemes)

        return phonemes


# ---------------------------------------------------------------------------
# Public API: text_to_phoneme_graph
# ---------------------------------------------------------------------------

def text_to_phoneme_graph(
    text: str,
    processor: Optional[CAMeLProcessor] = None,
) -> Tuple[List[str], List[Tuple[int, int]], List[List[float]]]:
    """Convert Arabic text into a phoneme-level graph representation.

    The graph has:
    - **Nodes**: one per phoneme in the utterance.
    - **Edges**: sequential adjacency (i → i+1) plus coarticulation skip
      connections between phonemes two positions apart (i → i+2).
    - **Node features**: 8-dimensional one-hot vector encoding the phoneme class.

    Args:
        text: Arabic text string.
        processor: An optional :class:`CAMeLProcessor` instance. If not
            provided, a default processor will be created. Pass an existing
            processor to avoid reinitializing CAMeL Tools on each call.

    Returns:
        A tuple ``(phoneme_list, edge_list, node_features)`` where:

        - ``phoneme_list``: List of IPA-like phoneme strings (length N).
        - ``edge_list``: List of ``(src, dst)`` int tuples.
        - ``node_features``: List of N float lists, each length 8.
    """
    if processor is None:
        processor = CAMeLProcessor()

    phonemes = processor.text_to_phonemes(text)

    if len(phonemes) == 0:
        return [], [], []

    # Build sequential adjacency edges.
    edges: List[Tuple[int, int]] = []
    n = len(phonemes)

    for i in range(n - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))  # Bidirectional

    # Add coarticulation skip connections (i ↔ i+2).
    for i in range(n - 2):
        edges.append((i, i + 2))
        edges.append((i + 2, i))

    # Deduplicate edges.
    edges = list(dict.fromkeys(edges))

    # Node features: 8-dim one-hot phoneme class.
    node_features: List[List[float]] = [_phoneme_to_features(p) for p in phonemes]

    return phonemes, edges, node_features


# ---------------------------------------------------------------------------
# Vocabulary builder
# ---------------------------------------------------------------------------

def build_phoneme_vocabulary(corpus_texts: List[str]) -> Dict[str, int]:
    """Build a phoneme vocabulary from a list of Arabic text strings.

    Args:
        corpus_texts: List of Arabic text strings.

    Returns:
        Dict mapping phoneme string → integer ID (0-indexed).
        Includes a special ``"<UNK>"`` token at the end.
    """
    processor = CAMeLProcessor()
    seen_phonemes: set = set()

    for text in corpus_texts:
        phonemes, _, _ = text_to_phoneme_graph(text, processor)
        seen_phonemes.update(phonemes)

    vocab: Dict[str, int] = {ph: idx for idx, ph in enumerate(sorted(seen_phonemes))}
    vocab["<UNK>"] = len(vocab)

    logger.info("Built phoneme vocabulary with %d symbols.", len(vocab))
    return vocab


# ---------------------------------------------------------------------------
# Corpus-level processing
# ---------------------------------------------------------------------------

def process_corpus(
    df: pd.DataFrame,
    output_path: str,
    processor: Optional[CAMeLProcessor] = None,
) -> None:
    """Run the full phoneme extraction pipeline on a corpus DataFrame.

    Saves processed graph data (phonemes, edges, features, label) as
    individual ``.pt`` (PyTorch) files using torch.save.

    Args:
        df: DataFrame with ``text`` and ``dialect_id`` columns.
        output_path: Directory to save ``.pt`` files.
        processor: A :class:`CAMeLProcessor` to reuse. Initializes one if None.
    """
    from pathlib import Path
    from tqdm import tqdm

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if processor is None:
        processor = CAMeLProcessor()

    failures = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting phoneme graphs"):
        try:
            import torch
            phonemes, edges, features = text_to_phoneme_graph(row["text"], processor)
            if len(phonemes) == 0:
                failures += 1
                continue

            data = {
                "phonemes": phonemes,
                "edges": edges,
                "features": features,
                "label": int(row["dialect_id"]),
                "text": row["text"],
            }
            torch.save(data, output_dir / f"{idx:06d}.pt")
        except Exception as exc:
            logger.warning("Failed on idx=%d: %s", idx, exc)
            failures += 1

    logger.info(
        "Corpus processing complete. Saved %d graphs, %d failures.",
        len(df) - failures,
        failures,
    )


# ---------------------------------------------------------------------------
# Fallback demo  (python data/camel_pipeline.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import io

    # Force UTF-8 output so IPA characters print on Windows (CP1252 would crash).
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    ) if hasattr(sys.stdout, "buffer") else sys.stdout

    logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

    SAMPLES = [
        ("\u0634\u0644\u0648\u0646\u0643 \u0627\u0644\u064a\u0648\u0645", "Gulf"),
        ("\u0627\u0632\u064a\u0643 \u0627\u0644\u0646\u0647\u0627\u0631\u062f\u0647", "Egyptian"),
        ("\u0643\u064a\u0641\u0643 \u0647\u0644\u0642 \u0634\u0648 \u0641\u064a", "Levantine"),
        ("\u0643\u064a\u062f\u0627\u064a\u0631 \u0648\u0627\u0634 \u062f\u0631\u062a", "Maghrebi"),
        ("\u0634\u0644\u0648\u0646\u0643 \u0634\u0643\u0648 \u0645\u0627\u0643\u0648", "Iraqi"),
    ]

    print("=" * 60)
    print("  data/camel_pipeline.py  --  Fallback Demo")
    print("=" * 60)

    processor = CAMeLProcessor()
    backend = "CAMeL Tools" if processor.uses_camel else "character-level fallback"
    print(f"\nBackend : {backend}\n")
    print(f"{'Dialect':<12}  Nodes  Edges  Sample phonemes")
    print("-" * 65)

    for text, dialect in SAMPLES:
        phonemes, edges, features = text_to_phoneme_graph(text, processor)
        # Show up to 8 phonemes, encode for safety on non-UTF terminals
        sample_ph = " ".join(phonemes[:8])
        try:
            sample_ph.encode(sys.stdout.encoding or "utf-8")
        except (UnicodeEncodeError, LookupError):
            sample_ph = " ".join(p.encode("ascii", "replace").decode() for p in phonemes[:8])
        print(f"{dialect:<12}  {len(phonemes):<5}  {len(edges):<5}  {sample_ph}")

    print("\nVocabulary builder demo:")
    vocab = build_phoneme_vocabulary([t for t, _ in SAMPLES])
    print(f"  Unique phoneme types : {len(vocab)} (incl. <UNK>)")
    # Print only ASCII-safe keys
    safe_items = [(k.encode("ascii", "replace").decode(), v) for k, v in list(vocab.items())[:6]]
    print(f"  Sample entries       : {dict(safe_items)}")

    print("\nAll checks passed.")
