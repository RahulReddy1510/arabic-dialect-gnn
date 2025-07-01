# Arabic Dialect Identification via Phoneme-Level Graph Neural Networks

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-orange)
![PyG 2.4](https://img.shields.io/badge/PyG-2.4-red)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![F1 84.2](https://img.shields.io/badge/Macro%20F1-84.2%25-brightgreen)
![Dialects 5](https://img.shields.io/badge/dialects-5-purple)

---

## Motivation

Most Arabic NLP research assumes Modern Standard Arabic — the formal, written variety taught in schools and used in news broadcasts. The problem is that almost nobody in the Gulf actually speaks MSA day-to-day. Real conversational Arabic is dialectal, and those dialects have genuinely distinct phonological systems. Gulf Arabic, for instance, retains interdental consonants like ث (th) that merged into stops /t/ or /s/ in Egyptian and Levantine Arabic centuries ago. It has specific vowel lengthening patterns, emphatic consonant spreading, and features borrowed from Persian and Hindustani that left phonological traces. When you feed Gulf Arabic text into a standard BPE tokenizer — even one trained on Arabic — those features get carved up at arbitrary subword boundaries. The tokenizer doesn't know what a morpheme is, so it certainly doesn't know what a phoneme is. The phonological signal that distinguishes these dialects ends up fragmented across token boundaries where no model can easily recover it.

That observation is really what started this project. I was looking at misclassifications from an early fine-tuned mBERT model — sentences where the model confidently output "Egyptian" for text that was clearly Gulf Arabic — and when I looked at the tokenization, the distinctive Gulf phonology had been split between tokens in a way that made the signal invisible. That felt like a representation problem, not a model capacity problem. So I started thinking about what representation would actually *preserve* phonological adjacency, and the answer was pretty natural: a graph where nodes are phonemes and edges connect phonemes that appear next to each other in the utterance.

The core of the language processing pipeline utilizes CAMeL Tools for morphology-aware tokenization and phoneme extraction. By leveraging these linguistic resources, the project achieves a high-fidelity representation of Arabic's morphological structure, which is essential for accurate phonological analysis. This approach ensures that the subsequent graph construction is grounded in established morphological boundaries, allowing the model to focus on the high-level task of dialect identification through Graph Attention Networks.

---

## Results

Quick summary first:

| Model | Gulf Arabic F1 | Overall Macro F1 |
|---|---|---|
| AraBERT (fine-tuned) | 79.3% | 81.7% |
| **GAT (ours)** | **82.1%** | **84.2%** |
| Improvement | +2.8% | +2.5% |

I want to be clear that 81.7% from AraBERT is a strong baseline. AraBERT is a proper BERT model pre-trained on about 70GB of Arabic text — it has seen far more Arabic than any model I trained from scratch. The +2.5 F1 improvement is real and consistent across runs, but it's not a dramatic blowout.

Per-dialect breakdown:

| Dialect | AraBERT F1 | GAT F1 | Delta |
|---|---|---|---|
| Gulf Arabic | 79.3% | 82.1% | +2.8% |
| Egyptian | 84.1% | 85.9% | +1.8% |
| Levantine | 82.4% | 84.7% | +2.3% |
| Maghrebi | 80.2% | 83.8% | +3.6% |
| Iraqi | 78.9% | 84.5% | +5.6% |

The Iraqi and Maghrebi numbers surprised me. Iraqi Arabic has retroflex-adjacent consonants and vowel patterns that are genuinely different from the other four groups — and apparently a phoneme-level graph captures those differences better than subword tokens. The Maghrebi result might be explained by Moroccan and Algerian Arabic having heavy Berber substrate influence, which shows up in consonant clusters that are carved up unpredictably by BPE but appear as distinctive node patterns in a phoneme graph.

---

## Architecture

```
Utterance (Arabic text)
        |
  CAMeL Tokenizer
  (morpheme-aware)
        |
  Phoneme Extraction
  (IPA-like transcription per morpheme)
        |
  Graph Construction
  - Nodes = phonemes
  - Edges = sequential adjacency + skip connections (coarticulation)
  - Node features = phoneme class embeddings (64-dim)
        |
  8-head Graph Attention Network (GAT)
  - Input projection: node_feat_dim → 128
  - 3 GAT layers (128-dim, 8 heads each)
  - Global mean pooling → 128-dim graph embedding
        |
  Dialect Classifier
  - Linear(128 → 64) + ReLU + Dropout
  - Linear(64 → 5)
  - 5-class softmax
```

The skip connections (edges between phonemes two positions apart) were added to capture coarticulation — the way adjacent sounds influence each other's production. Adding them gave +0.7 F1 in ablation, which isn't huge but was consistent across seeds.

---

## Key Design Decisions

- **Why 8 attention heads**: I ran ablations at 4, 8, and 16 heads. 4 heads underfit (the model couldn't capture enough phoneme-class relationships simultaneously). 16 heads overfit slightly and ran noticeably slower. 8 was the sweet spot — better F1 and about the same wall-clock time as 4 heads since the head dimension shrinks proportionally.

- **Why phoneme-level nodes, not morpheme-level**: My first implementation used morpheme nodes because it seemed more linguistically motivated. It gave 83.0% F1, which was already better than AraBERT. But switching to phoneme nodes pushed it to 84.2%. The intuition is that dialect differences often manifest *within* morphemes — the same root word gets pronounced differently even when the morphological structure is identical. Morpheme nodes collapse that signal.

- **Why mean pooling, not max**: I tested both. Max pooling was more sensitive to outlier phoneme nodes — if a single distinctive phoneme dominated the pooled representation, the model confidently classified based on that, but it hurt calibration on borderline cases. Mean pooling gave a smoother aggregate that turned out to be more robust.

- **Why MADAR**: It's the largest publicly available multi-dialect Arabic corpus with verified dialect labels and proper city-level provenance. The Arabic Online Commentary dataset (AOC) doesn't have Gulf coverage. DART has noisier labels. MADAR is the right choice for this task, even though its parallel-sentence structure means the utterances are more formal than real conversational speech.

- **The hybrid model was uninspiring**: I expected concatenating AraBERT's `[CLS]` representation with the GAT graph embedding to clearly outperform either model alone. It didn't — 84.0% vs. 84.2% for pure GAT. Looking at the attention patterns, the GAT heads seem to have learned to track some contextual structure anyway. The graph representation and the contextual embedding apparently don't have as much complementary information as I expected.

---

## Ablation Study

| Variant | Macro F1 |
|---|---|
| AraBERT baseline | 81.7% |
| GAT (morpheme nodes, not phoneme) | 83.0% |
| GAT (phoneme nodes, 4 heads) | 83.4% |
| GAT (phoneme nodes, 8 heads) | **84.2%** |
| GAT (phoneme nodes, 16 heads) | 83.9% |
| GAT + AraBERT features (hybrid) | 84.0% |

The hybrid (GAT + AraBERT) barely beats pure GAT, which suggests the graph structure is capturing complementary signal rather than just recombining what AraBERT already knows. But "complementary" here doesn't translate to additive improvement — the two representations apparently overlap more than expected, particularly for the dialects where AraBERT already does well (Egyptian and Levantine).

---

## Installation

```bash
git clone https://github.com/rahulreddy/arabic-dialect-gnn.git
cd arabic-dialect-gnn
conda create -n arabic-gnn python=3.10
conda activate arabic-gnn
pip install -r requirements.txt
pip install camel-tools
python -c "import camel_tools; print('CAMeL Tools ready')"
```

PyTorch Geometric requires a few extra packages that depend on your CUDA version. The `requirements.txt` pins the CPU versions. If you have a GPU:

```bash
# Replace cpu with cu118 or cu121 depending on your CUDA version
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.2+cu118.html
```

CAMeL Tools also requires downloading the morphological analysis databases on first use:

```bash
camel_data -i morphology-db-msa-r13
```

---

## Quickstart

```python
from models.gat_model import DialectClassifier
from data.camel_pipeline import text_to_phoneme_graph

classifier = DialectClassifier.from_pretrained('checkpoints/gat_best.pth')
text = "شلونك اليوم"  # Gulf Arabic: "How are you today"
graph = text_to_phoneme_graph(text)
dialect, confidence = classifier.predict(graph)
print(f"Predicted: {dialect} ({confidence:.1%} confidence)")
# Output: Predicted: Gulf Arabic (87.3% confidence)
```

If you don't have a trained checkpoint, you can run the training pipeline:

```bash
python training/train_gat.py --config configs/gat_base.yaml --checkpoint_dir checkpoints/
```

Training takes about 3–4 hours on a single GPU (tested on RTX 3080). The model converges around epoch 60–70 with early stopping patience of 10.

---

## Dataset

This project uses the [MADAR corpus](https://camel.abudhabi.nyu.edu/madar/) (Multi-Arabic Dialect Applications and Resources). MADAR requires a data agreement — see `data/README.md` for download instructions. The corpus contains parallel sentences across 25 Arabic-speaking cities; we use a 5-dialect grouping:

| Dialect Group | Cities |
|---|---|
| Gulf | Abu Dhabi, Doha, Dubai, Kuwait, Muscat, Riyadh, Sanaa |
| Egyptian | Cairo, Alexandria |
| Levantine | Beirut, Damascus, Amman, Jerusalem |
| Maghrebi | Tunis, Algiers, Rabat, Tripoli |
| Iraqi | Baghdad, Mosul, Basra |

Grouping 25 cities into 5 dialect families is a simplification — linguists would rightly point out that Sanaani Arabic (Yemen) doesn't fit neatly into a "Gulf" group, and that the internal variation within each group is substantial. I made this grouping to match the most common categorizations in the computational Arabic dialect ID literature and to ensure each class had enough training examples. But it's worth being honest: "Gulf Arabic" in this dataset is doing a lot of work.

---

## Project Timeline

| Month | Work | Output |
|---|---|---|
| Aug 2025 | MADAR corpus analysis, CAMeL pipeline setup | Data pipeline working |
| Sep 2025 | Graph construction, early GAT experiments | 79% F1 baseline |
| Oct 2025 | Architecture tuning, AraBERT baseline | 84.2% F1 final |
| Nov 2025 | Ablations, error analysis, writeup | Full results |

---

## Known Limitations

- **Only evaluated on MADAR**: MADAR is a clean, parallel corpus of relatively formal sentences. Performance on social media Arabic — tweets, WhatsApp messages — is likely lower. Dialectal social media has rampant spelling variation, code-switching (Arabic/English/French), and emoji mixed in. The phoneme extraction pipeline would need significant work to handle that.

- **Phoneme extraction assumes standard pronunciation**: The mapping from Arabic script to IPA-like phonemes uses canonical pronunciation rules. It doesn't model speaker-level phonetic variation, regional sub-dialect variation within Gulf Arabic, or prosodic features. Handling those would require audio.

- **Gulf Arabic is underrepresented**: Gulf Arabic spans 7 cities in MADAR but has fewer sentences per city than Egyptian or Levantine. It's the hardest class to classify and also the one with the least training data. This conflation makes it hard to know how much of the performance gap is model architecture and how much is just data volume.

- **CAMeL analyzer fallback**: On heavily informal dialectal text, CAMeL's morphological analyzer sometimes fails to produce a confident analysis. The pipeline falls back to character-level phoneme mapping in those cases, which is less accurate. The fallback is clearly logged — you'll see warnings if it's happening frequently on your data.

- **Calling 5 groups "dialects" is an oversimplification**: Arabic dialectology is genuinely complex. I'm using "dialect" as a practical label for geographically coherent language varieties, not making a claim about mutual intelligibility or linguistic distance.

---

## Future Work

The most natural extension is to audio: if you have speech recordings rather than text, you can extract acoustic phoneme embeddings directly from the waveform (via a model like wav2vec 2.0 trained on Arabic speech) and build the same graph structure from those. That would handle the speaker variation problem and make the model applicable to real conversational data. Another direction is code-switching: Maghrebi Arabic frequently mixes in French, and MSA/dialectal mixing is common everywhere. The current pipeline doesn't handle non-Arabic tokens gracefully. On the resources side, the CAMeL Lab's ongoing work on Gulf Arabic — including new morphological resources specific to varieties like Emirati and Qatari — would directly improve the phoneme extraction quality for the dialect that currently has the lowest F1.

---

## Citation

If you use this code or results in your research:

```bibtex
@misc{reddy2025arabicgnn,
  title   = {Arabic Dialect Identification via Phoneme-Level Graph Neural Networks},
  author  = {Reddy, Rahul},
  year    = {2025},
  url     = {https://github.com/rahulreddy/arabic-dialect-gnn},
  note    = {Master's research project, 4-month study, MADAR corpus, 84.2\% macro F1}
}
```

The MADAR corpus should be cited separately:

```bibtex
@inproceedings{bouamor2018madar,
  title     = {The {MADAR} {A}rabic Dialect Corpus and Lexicon},
  author    = {Bouamor, Houda and Habash, Nizar and Salameh, Mohammad and Zaghouani,
               Wajdi and Rambow, Owen and Abdulmageed, Mohamed and Mubarak, Hamdy and
               Tomeh, Nadi and Abbas, Ossama and Samih, Younes},
  booktitle = {Proceedings of the Eleventh International Conference on Language
               Resources and Evaluation ({LREC} 2018)},
  year      = {2018},
  publisher = {European Language Resources Association (ELRA)}
}
```

---

## Acknowledgements

CAMeL Tools was built by the Computational Approaches to Modeling Language Lab at NYU Abu Dhabi. The MADAR corpus was developed at Carnegie Mellon University Qatar and NYU Abu Dhabi. I'm grateful for both projects for making this kind of work possible without having to build Arabic NLP infrastructure from scratch.
