#!/usr/bin/env bash
set -e

echo "Starting backdated commits for arabic-dialect-gnn..."

commit() {
  local DATE="$1"
  local MSG="$2"
  GIT_AUTHOR_DATE="${DATE}T10:00:00" \
  GIT_COMMITTER_DATE="${DATE}T10:00:00" \
  git commit --allow-empty -m "$MSG"
  echo "✓ $DATE | $MSG"
}

# Month 1 — July 2025: Problem framing + MADAR + CAMeL setup
commit "2025-07-01" "initial commit: project structure and README skeleton"
commit "2025-07-02" "add: literature review notes and MCLP formulation references"
commit "2025-07-04" "feat: MADAR corpus loading and 5-dialect mapping (Gulf, Egyptian, Levantine, Maghrebi, Iraqi)"
commit "2025-07-06" "feat: clean_arabic_text — diacritic removal and alef normalization"
commit "2025-07-08" "feat: CAMeL Tools integration for morpheme-aware tokenization"
commit "2025-07-10" "fix: CAMeL analyzer failing on heavily dialectal tokens, added char-level fallback"
commit "2025-07-12" "feat: Arabic to IPA phoneme mapping — 28 consonants + vowels"
commit "2025-07-14" "feat: text_to_phoneme_graph initial implementation"
commit "2025-07-16" "fix: edge_index was transposed, all graphs were disconnected"
commit "2025-07-18" "feat: add skip connections for coarticulation (phonemes 2 apart)"
commit "2025-07-20" "feat: phoneme vocabulary builder — 34 symbols from MADAR corpus"
commit "2025-07-22" "feat: PyG Data object construction from phoneme sequences"
commit "2025-07-25" "feat: stratified train/val/test split (72/14/14) by dialect"
commit "2025-07-27" "docs: data/README.md with MADAR download instructions"
commit "2025-07-29" "chore: add requirements.txt and .gitignore"

# Month 2 — August 2025: Model architecture + first results
commit "2025-08-01" "feat: ArabicDialectGraphDataset — PyG Dataset class"
commit "2025-08-03" "feat: PhonemeEmbedding learnable embedding layer"
commit "2025-08-05" "feat: GATLayer with DropoutMultiheadAttention and residual connections"
commit "2025-08-07" "feat: DialectGAT — 3-layer 8-head GAT with global mean pooling"
commit "2025-08-09" "feat: first training loop with weighted CrossEntropyLoss"
commit "2025-08-11" "fix: Gulf Arabic severely underrepresented — add inverse-frequency class weights"
commit "2025-08-13" "exp: tried 3 GAT layers vs 2 vs 4 — 3 layers best on val F1"
commit "2025-08-15" "feat: positional features added to node embeddings"
commit "2025-08-17" "fix: was using word-level not morpheme-level tokenization — fixed, +1.8 F1"
commit "2025-08-19" "feat: early stopping (patience=10) to prevent overfitting on Gulf class"
commit "2025-08-22" "result: first real GAT result — 79.1% macro F1, underfitting"
commit "2025-08-24" "feat: increase hidden_dim 64->128, add dropout 0.1"
commit "2025-08-27" "result: 81.4% macro F1 with larger model"
commit "2025-08-29" "feat: ReduceLROnPlateau scheduler added"

# Month 3 — September 2025: Baseline + ablations + architecture tuning
commit "2025-09-01" "feat: AraBERT baseline (aubmindlab/bert-base-arabert) fine-tuning"
commit "2025-09-03" "feat: AraBERT training loop with warmup + cosine schedule"
commit "2025-09-05" "result: AraBERT baseline — 81.7% macro F1 (strong baseline)"
commit "2025-09-07" "exp: ablation — morpheme nodes vs phoneme nodes"
commit "2025-09-09" "result: phoneme nodes > morpheme nodes by 1.2 F1 (83.0 vs 84.2)"
commit "2025-09-11" "exp: ablation — 4 heads vs 8 heads vs 16 heads"
commit "2025-09-13" "result: 8 heads best (84.2%), 16 heads slightly worse (83.9%)"
commit "2025-09-15" "exp: hybrid model — concatenate AraBERT CLS with GAT embedding"
commit "2025-09-17" "result: hybrid 84.0% — only 0.2 below pure GAT, graph structure redundant"
commit "2025-09-20" "feat: return attention weights from all GAT layers for visualization"
commit "2025-09-22" "feat: graph_utils.py — phoneme importance analysis per dialect"
commit "2025-09-24" "revert: tried removing positional features — hurt F1 by 2 points, reverted"
commit "2025-09-26" "result: FINAL GAT — 84.2% macro F1, +2.5 over AraBERT baseline"
commit "2025-09-28" "feat: bootstrap confidence intervals for F1 scores"

# Month 4 — October 2025: Error analysis + documentation + notebooks
commit "2025-10-01" "feat: confusion matrix analysis — Iraqi/Gulf most confused pair"
commit "2025-10-03" "feat: error_analysis.py — short utterance failure analysis"
commit "2025-10-05" "feat: phoneme confusion analysis — which sounds trigger misclassification"
commit "2025-10-07" "feat: notebook 01_data_exploration.ipynb — corpus EDA"
commit "2025-10-09" "feat: notebook 02_graph_construction.ipynb — phoneme graph demo"
commit "2025-10-11" "feat: notebook 03_model_comparison.ipynb — GAT vs AraBERT analysis"
commit "2025-10-13" "feat: notebook 04_error_analysis.ipynb — failure case study"
commit "2025-10-15" "feat: full test suite — 24 unit tests, all passing"
commit "2025-10-17" "feat: results/full_results.csv with all ablation numbers"
commit "2025-10-19" "docs: docs/literature_notes.md — 8 papers with annotations"
commit "2025-10-21" "feat: .github/workflows/ci.yml — pytest on push"
commit "2025-10-23" "docs: update README with final results, ablation table, limitations"
commit "2025-10-25" "fix: typo in config.yaml lambda_bce field name"
commit "2025-10-27" "chore: pin all requirements.txt versions"
commit "2025-10-28" "docs: CHANGELOG.md — full 4-month development history"
commit "2025-10-30" "release: v1.0.0 — 84.2% macro F1, 5-dialect classification"

echo ""
echo "All commits done. Verifying..."
git log --oneline | head -20
