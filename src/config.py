"""
Assignment 3 - Visione Artificiale
Classificazione di Immagini Aeree tramite CNN Residuale Custom

Modulo: Configurazione centralizzata
"""

import os
from pathlib import Path

# ─── Path del progetto ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Creazione automatica delle directory di output
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Path dei dataset ─────────────────────────────────────────────────────────
AID_DIR = RAW_DIR / "AID"
UCMERCED_DIR = RAW_DIR / "UCMerced_LandUse" / "Images"

# ─── Path dei file generati ──────────────────────────────────────────────────
SPLITS_FILE = PROCESSED_DIR / "splits.npz"
PRETRAINED_MODEL = MODELS_DIR / "pretrained_aid.keras"
FINETUNED_MODEL = MODELS_DIR / "finetuned_ucmerced.keras"
SCRATCH_MODEL = MODELS_DIR / "scratch_ucmerced.keras"
BEST_MODEL = MODELS_DIR / "best_model.keras"

# ─── Classi UC Merced (21 classi, ordine alfabetico) ─────────────────────────
UCMERCED_CLASSES = [
    "agricultural", "airplane", "baseballdiamond", "beach", "buildings",
    "chaparral", "denseresidential", "forest", "freeway", "golfcourse",
    "harbor", "intersection", "mediumresidential", "mobilehomepark",
    "overpass", "parkinglot", "river", "runway", "sparseresidential",
    "storagetanks", "tenniscourt"
]
NUM_CLASSES_UCMERCED = len(UCMERCED_CLASSES)  # 21

# Classi AID (30 classi)
NUM_CLASSES_AID = 30

# ─── Iperparametri ───────────────────────────────────────────────────────────
IMG_SIZE = (128, 128)       # Dimensione di resize delle immagini
BATCH_SIZE = 32
RANDOM_STATE = 42           # Seed per riproducibilità

# Split UC Merced
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Split AID (pre-training)
AID_TRAIN_RATIO = 0.70
AID_VAL_RATIO = 0.30

# Training
PRETRAIN_LR = 1e-3          # Learning rate per pre-addestramento su AID
FINETUNE_LR = 1e-4          # Learning rate per fine-tuning su UC Merced
SCRATCH_LR = 1e-3           # Learning rate per addestramento da zero
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
