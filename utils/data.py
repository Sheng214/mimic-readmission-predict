"""Dataset loading + tokenisation helpers."""
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from .config import *

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

def _tokenise(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

def get_dataloaders(csv_path: str | Path):
    """Returns PyTorch DataLoaders for train & val splits."""
    # 1. Load CSV via 🤗 Datasets
    ds = load_dataset("csv", data_files=str(csv_path))["train"]

    # 2. Train/val stratified split using sklearn (gives reproducibility)
    train_idx, val_idx = train_test_split(
        list(range(len(ds))),
        test_size=0.2,
        stratify=ds["label"],
        random_state=SEED,
    )
    ds_train = ds.select(train_idx)
    ds_val   = ds.select(val_idx)

    # 3. Tokenise & set PyTorch format
    for split in (ds_train, ds_val):
        split = split.map(_tokenise, batched=True, desc="Tokenising")
        split = split.rename_column("label", "labels")
        split.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

    # 4. Create DataLoaders
    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val, batch_size=BATCH_SIZE, shuffle=False
    )
    return dl_train, dl_val