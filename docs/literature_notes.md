# Literature Notes

*Personal reading notes on the key papers that shaped this project. These are informal summaries for my own reference — not a formal literature review.*

---

## Core References

### Graph Attention Networks (GAT)
**Veličković et al., ICLR 2018** — [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)

The foundational paper. The core idea is computing attention weights between connected nodes, which generalizes the fixed-weight aggregation in standard GCN. The formulation:
```
α_ij = softmax( LeakyReLU( a^T [W h_i || W h_j] ) )
```
Multi-head attention reduces variance — the 8-head choice in this project comes directly from their ablation showing that heads=8 is the sweet spot before diminishing returns.

**What I took from it**: The return_attention_weights flag in PyG's GATConv makes it trivial to extract the learned attention patterns for visualization. This became the most useful debugging tool in the project.

---

### AraBERT
**Antoun et al., LREC 2020** — [arXiv:2003.00104](https://arxiv.org/abs/2003.00104)

BERT trained from scratch on ~70GB of Arabic text. The key preprocessing step is Farasa segmentation before tokenization, which dramatically improves MADAR performance over the original mBERT baseline. The paper reports 87.6% accuracy on MADAR 26-city (not the same as macro F1 on 5-dialect, so comparison is not straightforward).

**What I took from it**: The tokenization artifacts for dialectal text. *شلونك* (Gulf "how are you") gets tokenized as [##شلون, ##ك] or similar — the morpheme boundaries shift in dialectal text in ways the Arabic pre-training didn't fully expose the model to.

---

### MADAR Corpus
**Bouamor et al., LREC 2018** — [link](http://www.lrec-conf.org/proceedings/lrec2018/pdf/351.pdf)

200+ Arabic dialects (city-level), parallel sentences. The 26-city task is too fine-grained for current models; I used the 5-dialect grouping which maps MADAR cities to broader dialect regions (Gulf, Egyptian, Levantine, Maghrebi, Iraqi).

**What I took from it**: The class imbalance is a corpus collection artifact, not a reflection of real dialect prevalence. Egyptian Arabic has more web data, so Egyptian-origin sentences are over-represented.

---

### CAMeL Tools
**Obeid et al., 2020, LREC** — [GitHub](https://github.com/CAMeL-Lab/camel_tools)

The morphological analyzer that powers phoneme extraction. The `calima-msa-r13` database provides vowel pattern analysis that can be converted to approximate IPA. The character-level fallback in `data/camel_pipeline.py` was written after I realized many users won't have the LDC license for the full analyzer database.

**What I took from it**: The Arabic→phoneme mapping at the character level is surprisingly good for dialect identification purposes — the distinctive phonemes (ث/ذ/ظ retention, ق→ʔ, ج→g) are all at the character level in standard orthography.

---

## Supporting References

### Graph Neural Networks for NLP
**Yao et al., AAAI 2019** — Text GCN
- Constructed a graph where nodes are words AND documents, edges are word co-occurrence
- Showed that graph structure improves text classification
- **My adaptation**: using utterance-level graphs (one graph per sentence) rather than a corpus-level graph

### Phonological Dialect Features
**Habash, 2010** — Introduction to Arabic NLP (Book)
- Comprehensive overview of Arabic morphology and dialect variation
- Chapter 7 covers phonological differences between dialect groups
- Confirmed my intuitions about which phoneme contrasts are most dialectally distinctive

### Attention Weight Interpretability
**Jain & Wallace, NAACL 2019** — "Attention is not Explanation"
- Argued that attention weights don't necessarily reflect "what the model uses"
- Important caveat: the phoneme-level attention visualizations in Notebook 04 are for exploration, not causal claims
- The patterns are consistent and repeatable, but that's not the same as explanatory

---

## Ideas That Didn't Work

### Phoneme Bigram Features
*Tried in August 2025*

Added bigram phoneme features (concatenating adjacent phoneme embeddings) as node features. Hypothesis: the model would have explicit bigram context without needing attention.
- Result: 83.1% macro F1, slightly *worse* than the 8-dim one-hot encoding
- My interpretation: the GATConv attention mechanism already does this implicitly through the aggregation step. Adding it explicitly as a feature creates redundancy that hurts generalization.

### Self-Supervised Pre-training on Phoneme Graphs
*Considered in September 2025*

MADAR is small (~5K utterances per dialect in the training split). I considered pre-training the GAT on a larger unlabeled Arabic corpus by masking phoneme nodes (analogous to masked language modeling).
- Decision: not pursued due to time constraints and the diminishing likelihood of meaningful gains on 5-class classification (the architecture already converges well)
- This is probably worth doing for the 26-dialect task

---

## Key Decisions Revisited

| Decision | Alternatives Tried | Result |
|---|---|---|
| 8 attention heads | 4, 16 | 8 best (84.2%) |
| Mean pooling | Max pooling | Mean better (83.6% vs 84.2%) |
| 3 GAT layers | 2, 4 | 3 best; 4 overfit on small val |
| Phoneme nodes | Morpheme nodes | Phoneme: 84.2%, Morpheme: 83.0% |
| Sequential + skip-2 edges | Sequential only | +0.4 F1 from skip-2 edges |
| Weighted CE loss | Standard CE | +0.8 F1 on Gulf class |
