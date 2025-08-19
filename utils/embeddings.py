"""Attach ClinicalBERT CLS-token embeddings to a DataFrame.

Usage
-----
>>> from mimic_readmit_clinbert.embeddings import add_clinbert_embeddings
>>> df = add_clinbert_embeddings(df, text_col="lemmas")
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from .config import PRETRAINED_MODEL, MAX_LENGTH, BATCH_SIZE, DEVICE

# ────────────────────────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
model     = AutoModel.from_pretrained(PRETRAINED_MODEL).to(DEVICE)
model.eval()  # inference‑only
EMB_SIZE = model.config.hidden_size  # 768 for ClinicalBERT

@torch.no_grad()
def _embed_batch(texts: list[str]) -> torch.Tensor:
    """Return CLS embeddings for a list of texts (len ≤ BATCH_SIZE)."""
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(DEVICE)
    out = model(**enc)
    # CLS token is index 0 along sequence dimension
    return out.last_hidden_state[:, 0]  # shape (batch, EMB_SIZE)


def add_clinbert_embeddings(
    df: pd.DataFrame,
    *,
    text_col: str = "advanced_spacy_lemmas",
    batch_size: int = BATCH_SIZE,
    prefix: str = "emb_",
    progress: bool | Literal["tqdm"] = True,
) -> pd.DataFrame:
    """Return a **copy** of `df` with ClinicalBERT embedding columns appended.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `text_col` with cleaned lemmas.
    text_col : str, default "lemmas"
        Column to feed into ClinicalBERT.
    batch_size : int
        GPU/CPU batch size.
    prefix : str
        Column prefix for embedding dims → f"{prefix}{i}".
    progress : bool | "tqdm"
        Show a tqdm bar?
    """
    if text_col not in df.columns:
        raise KeyError(f"{text_col!r} column not found in DataFrame")

    texts = df[text_col].astype(str).apply(" ".join).tolist()
    n     = len(texts)
    n_batches = math.ceil(n / batch_size)
    bar = tqdm(total=n, desc="ClinicalBERT", disable=not progress)

    emb_list: list[torch.Tensor] = []
    for i in range(n_batches):
        batch_texts = texts[i * batch_size : (i + 1) * batch_size]
        emb = _embed_batch(batch_texts)
        emb_list.append(emb.cpu())
        bar.update(len(batch_texts))
    bar.close()

    # (n, EMB_SIZE) tensor → DataFrame with prefixed columns
    import numpy as np
    embs = torch.cat(emb_list).numpy()
    colnames = [f"{prefix}{i}" for i in range(EMB_SIZE)]
    df_emb   = pd.DataFrame(embs, columns=colnames, index=df.index)

    # Return original + embeddings (non‑destructive)
    return df_emb