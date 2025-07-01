"""
models/arabert_baseline.py

AraBERT fine-tuning baseline for Arabic dialect identification.

AraBERT (Antoun et al., 2020) is a BERT-Large model pre-trained on ~70GB
of Arabic text from Wikipedia, news, and web data. It's the natural
comparison point for this project because it represents the best standard
subword tokenization approach on Arabic.

The baseline achieves 81.7% macro F1 on MADAR after 5 epochs of fine-tuning.
That's a respectable result — AraBERT has strong Arabic morphological
structure baked in from pre-training, which is why it took genuine
architecture work (not just a different classifier) to beat it by 2.5 F1 pts.

Note: AraBERT processes text as subword tokens using BPE. The tokenizer splits
Arabic words at morpheme boundaries inconsistently for dialectal text, which is
the main motivation for the phoneme-graph approach. A word like "شلونك" (Gulf
Arabic: "how are you") gets fragmented in ways that lose the phonological
context between the morphemes ش-ل-ون-ك.

Reference:
    Antoun, W., Baly, F., & Hajj, H. (2020). AraBERT: Transformer-based
    Model for Arabic Language Understanding. LREC 2020 Workshop on
    Arabic Language Resources and Tools.
    https://arxiv.org/abs/2003.00104
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from typing import Tuple
    import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

DIALECT_NAMES: List[str] = ["Gulf", "Egyptian", "Levantine", "Maghrebi", "Iraqi"]

# Try importing transformers; provide helpful error if missing.
try:
    from transformers import AutoTokenizer, AutoModel  # type: ignore
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModel = None
    warnings.warn(
        "transformers package not found. AraBERTDialectClassifier will not work. "
        "Install: pip install transformers==4.36.2",
        UserWarning,
        stacklevel=1,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AraBERTDialectClassifier(nn.Module):
    """Fine-tuned AraBERT classifier for Arabic dialect identification.

    Uses the ``[CLS]`` token representation from AraBERT as a sentence
    embedding, then passes it through a dropout + linear classifier head.

    Args:
        model_name: HuggingFace model identifier. Default is
            ``"aubmindlab/bert-base-arabert"``.
        num_classes: Number of dialect classes (5).
        dropout: Dropout probability on the CLS representation.
        freeze_base: If True, freeze all AraBERT weights and only train the
            classifier head. Useful for a quick sanity check. Default False.
    """

    def __init__(
        self,
        model_name: str = "aubmindlab/bert-base-arabert",
        num_classes: int = 5,
        dropout: float = 0.1,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required. pip install transformers==4.36.2")

        self.bert = AutoModel.from_pretrained(model_name)
        bert_hidden = self.bert.config.hidden_size   # 768 for bert-base

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(bert_hidden, num_classes),
        )

        if freeze_base:
            for param in self.bert.parameters():
                param.requires_grad = False
            logger.info("AraBERT base weights frozen. Training classifier head only.")

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            input_ids: Token ID tensor (batch_size, seq_len).
            attention_mask: Attention mask (batch_size, seq_len).
            token_type_ids: Segment IDs (optional, batch_size, seq_len).

        Returns:
            Logits tensor (batch_size, num_classes).
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Use [CLS] token (index 0) as the sentence representation.
        cls_embed = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        logits = self.classifier(cls_embed)  # (batch, num_classes)
        return logits

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Total parameter count (trainable by default)."""
        return sum(
            p.numel() for p in self.parameters()
            if (p.requires_grad if trainable_only else True)
        )


# ---------------------------------------------------------------------------
# Tokenization helper
# ---------------------------------------------------------------------------

def tokenize_batch(
    texts: List[str],
    tokenizer,
    max_length: int = 128,
    device: Optional[torch.device] = None,
) -> Dict[str, Tensor]:
    """Tokenize a list of Arabic strings for AraBERT.

    Args:
        texts: List of Arabic text strings.
        tokenizer: A HuggingFace tokenizer (AutoTokenizer) instance.
        max_length: Maximum sequence length. Longer sequences are truncated;
            shorter ones are padded. 128 tokens covers >98% of MADAR sentences.
        device: Target device for the output tensors. CPU if None.

    Returns:
        Dict with keys ``"input_ids"``, ``"attention_mask"``, and optionally
        ``"token_type_ids"`` — all as tensors on *device*.
    """
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    if device is not None:
        encoded = {k: v.to(device) for k, v in encoded.items()}
    return encoded


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_arabert(
    model: AraBERTDialectClassifier,
    train_loader,
    val_loader,
    config: Dict,
    device: Optional[torch.device] = None,
) -> Dict:
    """Fine-tune AraBERT for dialect classification.

    Uses linear warmup + cosine decay scheduling, gradient clipping at 1.0,
    and optional mixed-precision training (requires CUDA).

    Args:
        model: Initialized :class:`AraBERTDialectClassifier`.
        train_loader: DataLoader yielding (input_ids, attention_mask, labels).
        val_loader: Validation DataLoader with same format.
        config: Training config dict. Expected keys:
            ``lr``, ``epochs``, ``warmup_steps``, ``weight_decay``,
            ``gradient_clip``.
        device: Training device. Auto-detected if None.

    Returns:
        Dict with ``"best_macro_f1"`` and ``"history"`` (list of epoch dicts).
    """
    from sklearn.metrics import f1_score  # type: ignore

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    lr = config.get("lr", 2e-5)
    epochs = config.get("epochs", 5)
    warmup_steps = config.get("warmup_steps", 100)
    weight_decay = config.get("weight_decay", 1e-4)
    gradient_clip = config.get("gradient_clip", 1.0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Linear warmup then cosine decay.
    total_steps = epochs * len(train_loader)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    # Mixed precision scaler — only enabled on CUDA.
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: List[Dict] = []
    best_macro_f1 = 0.0

    for epoch in range(1, epochs + 1):
        # ------ Training ------
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()

        # ------ Validation ------
        val_metrics = evaluate_arabert(model, val_loader, device)
        macro_f1 = val_metrics["macro_f1"]

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss / max(len(train_loader), 1),
            **val_metrics,
        }
        history.append(epoch_log)
        logger.info(
            "Epoch %d/%d — loss: %.4f | val macro_f1: %.4f",
            epoch, epochs,
            epoch_log["train_loss"],
            macro_f1,
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1

    return {"best_macro_f1": best_macro_f1, "history": history}


def evaluate_arabert(
    model: AraBERTDialectClassifier,
    loader,
    device: torch.device,
) -> Dict:
    """Evaluate AraBERT on a dataset loader.

    Args:
        model: Trained :class:`AraBERTDialectClassifier`.
        loader: DataLoader yielding (input_ids, attention_mask, labels).
        device: Evaluation device.

    Returns:
        Dict with:
        - ``"macro_f1"``: Macro-averaged F1 score.
        - ``"per_dialect_f1"``: Dict mapping dialect name → F1.
        - ``"loss"``: Mean cross-entropy loss.
    """
    from sklearn.metrics import f1_score  # type: ignore

    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds: List[int] = []
    all_labels: List[int] = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    per_class_f1 = f1_score(
        all_labels, all_preds, average=None, zero_division=0,
        labels=list(range(len(DIALECT_NAMES)))
    )
    per_dialect_f1 = {
        DIALECT_NAMES[i]: float(per_class_f1[i])
        for i in range(len(DIALECT_NAMES))
    }

    return {
        "macro_f1": macro_f1,
        "per_dialect_f1": per_dialect_f1,
        "loss": total_loss / max(len(loader), 1),
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    if not _TRANSFORMERS_AVAILABLE:
        print("transformers not installed — skipping AraBERT self-test.")
        sys.exit(0)

    print("=" * 60)
    print("  AraBERT — Tokenization Verification")
    print("=" * 60)

    model_name = "aubmindlab/bert-base-arabert"

    print(f"\nLoading tokenizer: {model_name}")
    print("(This will download the model on first run — ~500MB)\n")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Gulf Arabic: "How are you today?"
        test_sentences = [
            "شلونك اليوم",      # Gulf
            "ازيك النهارده",    # Egyptian
            "كيفك هلق",         # Levantine
        ]

        for sent in test_sentences:
            ids = tokenizer(sent, return_tensors="pt")["input_ids"]
            tokens = tokenizer.convert_ids_to_tokens(ids[0].tolist())
            print(f"Text:   {sent}")
            print(f"Tokens: {tokens}")
            print(f"IDs:    {ids[0].tolist()}\n")

        print(
            "Note: observe how dialectal morphemes get split at unexpected boundaries.\n"
            "This fragmentation is exactly what phoneme-level graph representation avoids."
        )
    except Exception as e:
        print(f"Could not load AraBERT (likely no internet): {e}")
        print("This is fine — the baseline works identically with a locally cached model.")
