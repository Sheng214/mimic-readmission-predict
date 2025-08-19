"""Central configuration.
Edit paths & hyper-params here only!"""
from pathlib import Path

# HuggingFace model hub id
PRETRAINED_MODEL = "emilyalsentzer/Bio_ClinicalBERT"

# Embedding params
MAX_LENGTH   = 128
BATCH_SIZE   = 16

# Device helper – overridden automatically if CUDA visible
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")