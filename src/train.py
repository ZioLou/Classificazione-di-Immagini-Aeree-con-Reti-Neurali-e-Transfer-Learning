"""
Modulo: Training
- Pre-addestramento su AID (Strategia 1, fase 1)
- Fine-tuning su UC Merced (Strategia 1, fase 2)
- Addestramento da zero su UC Merced (Strategia 2)
- Early Stopping + salvataggio modello migliore in tutti gli esperimenti
"""

import numpy as np
import cv2
import json
import tensorflow as tf
from tensorflow import keras
from src.config import (
    PRETRAINED_MODEL, FINETUNED_MODEL, SCRATCH_MODEL, PROCESSED_DIR,
    NUM_CLASSES_AID, NUM_CLASSES_UCMERCED,
    PRETRAIN_LR, FINETUNE_LR, SCRATCH_LR,
    MAX_EPOCHS, EARLY_STOPPING_PATIENCE, BATCH_SIZE
)
from src.data_loader import (
    load_aid_split, load_ucmerced_split, load_images_from_paths
)
from src.model import build_model


# ──────────────────────────────────────────────────────────────────────────────
# Data Augmentation con OpenCV (applicata SOLO al training set)
# ──────────────────────────────────────────────────────────────────────────────

def augment_image(img):
    # Augmentation leggera per immagini aeree.
    # Le immagini satellitari sono invarianti al flip orizzontale.

    # Flip orizzontale casuale (50% probabilità)
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)

    # Lieve variazione di luminosità (+/- 10%)
    factor = np.random.uniform(0.9, 1.1)
    img = np.clip(img * factor, 0.0, 1.0).astype(np.float32)

    return img


# ──────────────────────────────────────────────────────────────────────────────
# Generator per training con augmentation on-the-fly
# ──────────────────────────────────────────────────────────────────────────────

def train_generator(X, y, batch_size, augment=False):
    # Generator infinito: produce batch (immagini, label).
    # Ad ogni epoca i dati vengono rimescolati per evitare pattern fissi.

    n = len(X)
    while True:
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:min(start + batch_size, n)]
            batch_X = X[batch_idx].copy()
            batch_y = y[batch_idx]

            # Augmentation applicata solo se richiesto
            if augment:
                for i in range(len(batch_X)):
                    batch_X[i] = augment_image(batch_X[i])

            yield batch_X, batch_y


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks comuni a tutti gli esperimenti
# ──────────────────────────────────────────────────────────────────────────────

def get_callbacks(model_path):
    # Early Stopping: ferma il training se val_loss non migliora per N epoche.
    # ModelCheckpoint: salva automaticamente il modello con la migliore val_loss.

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    return [early_stop, checkpoint]


# ──────────────────────────────────────────────────────────────────────────────
# Salvataggio history (loss/accuracy) per i plot successivi
# ──────────────────────────────────────────────────────────────────────────────

def save_history(history, filepath):
    # Converte la history di Keras in un dizionario JSON-serializzabile
    hist_dict = {}
    for key, values in history.history.items():
        hist_dict[key] = [float(v) for v in values]
    with open(filepath, 'w') as f:
        json.dump(hist_dict, f, indent=2)
    print(f"  History salvata in: {filepath}")


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGIA 1 — Fase 1: Pre-addestramento su AID (30 classi, circa 10k immagini)
# ══════════════════════════════════════════════════════════════════════════════

def pretrain_on_aid():
    print("\n" + "=" * 60)
    print("STRATEGIA 1 — Fase 1: Pre-addestramento su AID")
    print("=" * 60)

    # Carica e splitta il dataset AID (70% train, 30% val)
    tr_p, tr_l, v_p, v_l, class_names = load_aid_split()

    print("\nCaricamento immagini AID training...")
    X_train, y_train = load_images_from_paths(tr_p, tr_l, class_names)
    print("Caricamento immagini AID validation...")
    X_val, y_val = load_images_from_paths(v_p, v_l, class_names)

    # Codifica one-hot delle label (30 classi)
    y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES_AID)
    y_val_cat = keras.utils.to_categorical(y_val, NUM_CLASSES_AID)

    # Costruisce il modello con 30 classi per AID
    model = build_model(NUM_CLASSES_AID)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=PRETRAIN_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"\nParametri totali: {model.count_params():,}")

    # Training con generator (augmentation on-the-fly sul training)
    callbacks = get_callbacks(str(PRETRAINED_MODEL))
    steps = int(np.ceil(len(X_train) / BATCH_SIZE))

    history = model.fit(
        train_generator(X_train, y_train_cat, BATCH_SIZE, augment=True),
        steps_per_epoch=steps,
        validation_data=(X_val, y_val_cat),
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    save_history(history, str(PROCESSED_DIR / "history_pretrain_aid.json"))
    print(f"\nModello pre-addestrato salvato in: {PRETRAINED_MODEL}")
    return model, history


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGIA 1 — Fase 2: Fine-tuning su UC Merced (21 classi)
# ══════════════════════════════════════════════════════════════════════════════

def finetune_on_ucmerced():
    print("\n" + "=" * 60)
    print("STRATEGIA 1 — Fase 2: Fine-tuning su UC Merced")
    print("=" * 60)

    # Carica i dati UC Merced con lo split salvato
    tr_p, tr_l, v_p, v_l, _, _, class_names = load_ucmerced_split()

    print("\nCaricamento immagini UCMerced...")
    X_train, y_train = load_images_from_paths(tr_p, tr_l, class_names)
    X_val, y_val = load_images_from_paths(v_p, v_l, class_names)

    y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES_UCMERCED)
    y_val_cat = keras.utils.to_categorical(y_val, NUM_CLASSES_UCMERCED)

    # Carica il modello pre-addestrato su AID (30 classi)
    print(f"\nCaricamento modello pre-addestrato da: {PRETRAINED_MODEL}")
    pretrained = keras.models.load_model(str(PRETRAINED_MODEL))

    # Sostituisce l'ultimo layer: Dense(30 classi AID) → Dense(21 classi UC Merced)
    # layers[-1] = Dense(30), layers[-2] = Dropout → prendiamo l'output del Dropout
    x = pretrained.layers[-2].output
    outputs = keras.layers.Dense(
        NUM_CLASSES_UCMERCED, activation='softmax', name='output_ucmerced'
    )(x)
    model = keras.Model(pretrained.input, outputs)

    # ─── Strategia B: congela parziale ────────────────────────────────────────
    # Congela stem + stage 1 + stage 2 (feature generali già apprese su AID).
    # Lascia trainabili stage 3 (128 filtri) + MLP per adattarsi a UC Merced.

    # Prima congela tutto
    for layer in model.layers:
        layer.trainable = False

    # Poi sblocca da stage 3 in poi (primo Conv2D con 128 filtri)
    found_stage3 = False
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D) and layer.filters == 128:
            found_stage3 = True
        if found_stage3:
            layer.trainable = True

    # Riepilogo parametri trainabili
    total_params = model.count_params()
    trainable_params = sum(w.numpy().size for w in model.trainable_weights)
    frozen_params = total_params - trainable_params
    print(f"  Parametri totali:     {total_params:,}")
    print(f"  Parametri trainabili: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"  Parametri congelati:  {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")

    # LR più basso del pre-training per non rovinare i pesi appresi
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINETUNE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Training con early stopping
    callbacks = get_callbacks(str(FINETUNED_MODEL))
    steps = int(np.ceil(len(X_train) / BATCH_SIZE))

    history = model.fit(
        train_generator(X_train, y_train_cat, BATCH_SIZE, augment=True),
        steps_per_epoch=steps,
        validation_data=(X_val, y_val_cat),
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    save_history(history, str(PROCESSED_DIR / "history_finetune.json"))
    print(f"\nModello fine-tuned salvato in: {FINETUNED_MODEL}")
    return model, history


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGIA 2: Addestramento da zero su UC Merced (senza pre-training)
# ══════════════════════════════════════════════════════════════════════════════

def train_from_scratch():
    print("\n" + "=" * 60)
    print("STRATEGIA 2: Addestramento da zero su UC Merced")
    print("=" * 60)

    # Carica gli STESSI dati UC Merced con lo STESSO split della strategia 1
    tr_p, tr_l, v_p, v_l, _, _, class_names = load_ucmerced_split()

    print("\nCaricamento immagini UCMerced...")
    X_train, y_train = load_images_from_paths(tr_p, tr_l, class_names)
    X_val, y_val = load_images_from_paths(v_p, v_l, class_names)

    y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES_UCMERCED)
    y_val_cat = keras.utils.to_categorical(y_val, NUM_CLASSES_UCMERCED)

    # Modello fresco con pesi inizializzati casualmente (21 classi)
    model = build_model(NUM_CLASSES_UCMERCED)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=SCRATCH_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"\nParametri totali: {model.count_params():,} (tutti trainabili)")

    # Training con early stopping
    callbacks = get_callbacks(str(SCRATCH_MODEL))
    steps = int(np.ceil(len(X_train) / BATCH_SIZE))

    history = model.fit(
        train_generator(X_train, y_train_cat, BATCH_SIZE, augment=True),
        steps_per_epoch=steps,
        validation_data=(X_val, y_val_cat),
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    save_history(history, str(PROCESSED_DIR / "history_scratch.json"))
    print(f"\nModello from-scratch salvato in: {SCRATCH_MODEL}")
    return model, history
