"""
Modulo: Caricamento e pre-processing dati
- Caricamento immagini con OpenCV (obbligatorio per l'assignment)
- Resize e normalizzazione [0,1]
- Split stratificato UC Merced (70/15/15) salvato su disco
- Split AID per pre-training (70/30)
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import (
    AID_DIR, UCMERCED_DIR, SPLITS_FILE, IMG_SIZE,
    TRAIN_RATIO, VAL_RATIO, RANDOM_STATE,
    AID_TRAIN_RATIO, BATCH_SIZE
)


# ──────────────────────────────────────────────────────────────────────────────
# Caricamento e pre-processing di una singola immagine
# ──────────────────────────────────────────────────────────────────────────────

def load_image(path, target_size=IMG_SIZE):

    #Carica un'immagine, converte in RGB, ridimensiona e normalizza.
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Impossibile caricare l'immagine: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize alla dimensione target (vedere config.py)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Normalizzazione pixel da [0,255] a [0,1] per stabilità del training
    img = img.astype(np.float32) / 255.0
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Scansione di un dataset organizzato in sottocartelle (una per classe)
# ──────────────────────────────────────────────────────────────────────────────

def scan_dataset(dataset_dir):
    # Raccoglie i path delle immagini e le label (nome cartella = classe).
    # Restituisce: lista di path (stringhe), lista di label (stringhe), lista classi ordinate.

    paths = []
    labels = []

    # Ogni sottocartella è una classe; ordiniamo per avere indici deterministici
    class_names = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith(".")
    ])

    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        # Raccogliamo tutti i file immagine nella cartella della classe
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                paths.append(os.path.join(class_dir, fname))
                labels.append(class_name)

    print(f"  Dataset: {dataset_dir}")
    print(f"  Trovate {len(paths)} immagini in {len(class_names)} classi")
    return paths, labels, class_names


# ──────────────────────────────────────────────────────────────────────────────
# Split stratificato UC Merced (70/15/15) — salvato su disco
# ──────────────────────────────────────────────────────────────────────────────

def create_ucmerced_split():
    # Crea lo split stratificato 70/15/15 per UC Merced e lo salva in splits.npz.
    # Lo split viene fatto UNA SOLA VOLTA e riusato per entrambe le strategie.
    
    print("\n[SPLIT] Creazione split UC Merced...")
    paths, labels, class_names = scan_dataset(str(UCMERCED_DIR))

    paths = np.array(paths)
    labels = np.array(labels)

    # Primo split: separiamo il test set (15%)
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels,
        test_size=0.15,
        stratify=labels,
        random_state=RANDOM_STATE
    )

    # Secondo split: dal restante 85% separiamo la validazione
    # 15% del totale = 15/85 ≈ 0.1765 del sottoinsieme train+val
    val_fraction = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_fraction,
        stratify=train_val_labels,
        random_state=RANDOM_STATE
    )

    # Salvataggio su disco per riproducibilità tra esperimenti
    np.savez(
        str(SPLITS_FILE),
        train_paths=train_paths, train_labels=train_labels,
        val_paths=val_paths, val_labels=val_labels,
        test_paths=test_paths, test_labels=test_labels,
        class_names=np.array(class_names)
    )

    print(f"  Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
    print(f"  Split salvato in: {SPLITS_FILE}")
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names


def load_ucmerced_split():
    # Carica lo split salvato da disco. Se non esiste, lo crea.

    if SPLITS_FILE.exists():
        print("\n[SPLIT] Caricamento split UC Merced da disco...")
        data = np.load(str(SPLITS_FILE), allow_pickle=True)
        train_paths = data["train_paths"]
        train_labels = data["train_labels"]
        val_paths = data["val_paths"]
        val_labels = data["val_labels"]
        test_paths = data["test_paths"]
        test_labels = data["test_labels"]
        class_names = list(data["class_names"])
        print(f"  Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
        return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names
    else:
        # Se lo split non esiste ancora, lo creiamo
        return create_ucmerced_split()


# ──────────────────────────────────────────────────────────────────────────────
# Split AID per pre-training (70/30)
# ──────────────────────────────────────────────────────────────────────────────

def load_aid_split():
    # Crea lo split 70/30 per il dataset AID (pre-training).
    # Non serve salvarlo su disco perché viene usato una sola volta.

    print("\n[SPLIT] Creazione split AID per pre-training...")
    paths, labels, class_names = scan_dataset(str(AID_DIR))

    paths = np.array(paths)
    labels = np.array(labels)

    # Split semplice: 70% training, 30% validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels,
        test_size=1 - AID_TRAIN_RATIO,  # 0.30 va in validation
        stratify=labels,
        random_state=RANDOM_STATE
    )

    print(f"  Train: {len(train_paths)} | Val: {len(val_paths)}")
    return train_paths, train_labels, val_paths, val_labels, class_names


# ──────────────────────────────────────────────────────────────────────────────
# Caricamento batch di immagini in memoria (array numpy)
# ──────────────────────────────────────────────────────────────────────────────

def load_images_from_paths(paths, labels, class_names):
    # Carica tutte le immagini dai path e converte le label in indici numerici.
    # Restituisce: X (N, H, W, 3) float32, y (N,) int.

    # Mappiamo ogni nome di classe al suo indice numerico
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    X = np.zeros((len(paths), IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.float32)
    y = np.zeros(len(paths), dtype=np.int32)

    for i, (path, label) in enumerate(zip(paths, labels)):
        X[i] = load_image(path)
        y[i] = class_to_idx[label]

    print(f"  Caricate {len(X)} immagini — shape: {X.shape}")
    return X, y
