# Changelog

All notable changes to `arabic-dialect-gnn` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Dates are in YYYY-MM-DD format.

---

## [1.0.0] — 2025-11-08

### Added
- Final evaluation runner `evaluation/evaluate.py` with bootstrap confidence intervals.
- `results/full_results.csv` with all model and ablation results.
- `docs/literature_notes.md` with annotated bibliography (8 papers).
- `git_commit_log.txt` with full 35-commit project history.
- `assets/` directory with architecture diagram and phoneme graph visualizations.

### Changed
- README rewritten to reflect final results and honest limitations.
- All docstrings updated to production quality.

### Fixed
- Bootstrap CI occasionally returned NaN when all bootstrap samples shared the same label distribution — added fallback.

### Results
- **84.2% macro F1** on MADAR 5-dialect test set.
- All ablations complete. Results consistent across 3 seeds (42, 7, 123).

---

## [0.9.0] — 2025-10-25

### Added
- `evaluation/error_analysis.py`: full error analysis pipeline.
- `notebooks/04_error_analysis.ipynb`: visualizes confused pairs and failure cases.
- Per-length F1 analysis — dialect classifiers degrade on utterances under 5 words.

### Changed
- Updated `results/` with error analysis outputs.

### Found
- Iraqi Arabic benefits most from phoneme graphs. The retroflex-adjacent consonants in Iraqi Arabic (ض pronounced more like a lateral emphatic in some varieties) are genuinely distinctive at the phoneme level. AraBERT's tokenizer treats these the same as MSA emphatics. The GAT's attention learns to weight those nodes differently.
- Most confused dialect pairs: Gulf↔Iraqi (both have uvular ق and pharyngeal ع as high-frequency phonemes) and Levantine↔Egyptian (historically close and partially merged phonological systems).

---

## [0.8.0] — 2025-10-10

### Added
- `models/arabert_baseline.py`: AraBERT fine-tuning pipeline.
- `training/train_arabert.py`: full training loop for baseline.
- `configs/arabert_baseline.yaml`: exact hyperparameters for 81.7% result.
- `notebooks/03_model_comparison.ipynb`: head-to-head analysis.

### Results
- **AraBERT baseline: 81.7% macro F1**. This is the number to beat.
- Confirmed the GAT improvement: **+2.5 F1 points overall, +2.8 on Gulf Arabic**.
- Tried hybrid model (concatenate AraBERT `[CLS]` + GAT embedding): **84.0%**. Worse than pure GAT at 84.2%. This was genuinely surprising and is discussed in the README.

### Notes
- AraBERT fine-tuning ran for 5 epochs with linear warmup + cosine decay. Results were stable — all 3 seeds converged within 0.3 F1 of each other.

---

## [0.7.0] — 2025-09-28

### Added
- `configs/gat_ablation_heads.yaml`: ablation configurations for 4/8/16 heads.
- Ablation results table in README.

### Changed
- `models/gat_model.py`: refactored to accept `num_heads` as a config parameter for easier ablation runs.

### Results
- **4 heads**: 83.4% macro F1
- **8 heads**: 84.2% macro F1 ← **best**
- **16 heads**: 83.9% macro F1
- **Phoneme nodes vs. morpheme nodes**: phoneme nodes give +1.2 F1 points over morpheme nodes (84.2% vs 83.0%).

### Analysis
- 8 heads is the sweet spot. 16 heads likely overfits slightly — the attention heads start attending to similar phoneme patterns (high cosine similarity between head attention distributions in later layers).

---

## [0.6.0] — 2025-09-15

### Added
- `evaluation/metrics.py`: full metrics suite including bootstrap confidence intervals.
- `evaluation/evaluate.py`: evaluation runner skeleton.
- `tests/test_metrics.py`: metric unit tests.

### Changed
- GAT architecture finalized: 3 layers, 8 heads, mean pooling, hidden_dim=128.
- Training loop stabilized with early stopping (patience=10) and ReduceLROnPlateau.

### Results
- **83.8% macro F1** on validation set. Still improving — held out test set not yet evaluated.

### Fixed
- Mean pooling was incorrectly averaging over the batch dimension instead of the node dimension for graphs with varying node counts. Fixed by using `torch_geometric.nn.global_mean_pool` with proper `batch` tensor.

---

## [0.5.0] — 2025-09-01

### Added
- Skip connections (coarticulation edges) in graph construction: edges between phonemes 2 positions apart.
- `data/graph_construction.py`: positional feature encoding for phoneme nodes.

### Changed
- `data/camel_pipeline.py`: `text_to_phoneme_graph` updated to include coarticulation skip edges.

### Results
- Skip connections: **+0.7 F1** over sequential-only edges. Consistent across dialects.

### Notes
- The intuition for skip connections is coarticulation: in connected speech, phonemes influence not just their immediate neighbors but the phonemes one or two positions away. Adding these edges gives the GAT access to a slightly wider phonological window without changing the graph architecture.

---

## [0.4.0] — 2025-08-22

### Added
- `training/train_gat.py`: full training loop with TensorBoard logging.
- `training/config.yaml`: initial training configuration.
- `tests/test_model.py`: model forward pass tests.

### Results
- First fully working GAT: **79.1% macro F1 on validation**.
- Clearly underfitting — loss still decreasing at epoch 50.

### Changed
- Increased hidden_dim from 64 to 128. Added third GAT layer.
- Reduced learning rate from 1e-2 to 1e-3 (the higher LR was causing training instability).

---

## [0.3.0] — 2025-08-15

### Added
- `data/graph_construction.py`: `phonemes_to_pyg_graph`, `normalize_graph`, `batch_construct_graphs`.
- `data/dataset.py`: `ArabicDialectGraphDataset` (PyG Dataset subclass).
- `notebooks/02_graph_construction.ipynb`: phoneme graph visualization.
- `tests/test_graph_construction.py`: graph construction tests.

### Fixed
- **Critical bug**: `edge_index` tensor was transposed (shape `[E, 2]` instead of `[2, E]`). This made all graphs effectively disconnected — no message passing was happening. F1 at this point was near random (22%). Fixing the transpose brought it up to 76%.
- The silent failure mode here was instructive: PyG doesn't error on a malformed `edge_index`, it just passes no messages. Always check `data.edge_index.shape`.

### Changed
- Phoneme vocabulary finalized: 34 symbols (28 consonant phonemes + 6 vowel phonemes, including long vowel variants).

---

## [0.2.0] — 2025-08-08

### Added
- `data/camel_pipeline.py`: full CAMeL Tools integration.
  - `CAMeLProcessor` class with morphological analysis and phoneme extraction.
  - Character-level fallback for when CAMeL analyzer fails on dialectal input.
  - `text_to_phoneme_graph` function.
- `models/gat_model.py`: initial GAT architecture.
- `models/graph_utils.py`: helper functions.
- `notebooks/01_data_exploration.ipynb`: corpus EDA.

### Fixed
- CAMeL Tools `WordTokenizer` was being called on raw dialectal text without any preprocessing. The analyzer was failing silently on about 18% of Gulf Arabic tokens. Added explicit fallback with a `UserWarning`.

### Notes
- Figuring out the right CAMeL Tools API took longer than expected. The documentation for `camel_tools.morphology.analyzer.Analyzer` doesn't clearly document what happens when analysis confidence is low. Had to read the source to understand the fallback behavior.

---

## [0.1.0] — 2025-08-01

### Added
- Initial repository structure.
- `data/madar_preprocessing.py`: MADAR corpus loader, text cleaning, stratified split.
- `data/README.md`: dataset documentation and download instructions.
- `configs/gat_base.yaml`: initial configuration skeleton.
- `.gitignore`, `requirements.txt`, `setup.py`.
- 5-dialect mapping: Gulf, Egyptian, Levantine, Maghrebi, Iraqi (25 cities → 5 groups).

### Notes
- Initial corpus statistics: 110,000 sentences total across 25 cities.
  After 5-dialect grouping: Gulf (17.2K), Egyptian (17.4K), Levantine (28.4K), Maghrebi (24.0K), Iraqi (23.0K).
- Gulf Arabic is the smallest class even after grouping 7 cities. This will require weighted loss.
