"""
Modulo: Definizione architettura CNN residuale custom
- Blocco residuale (Conv-BN-ReLU-Conv-BN + skip connection)
- 3 blocchi residuali con downsampling progressivo (32 → 64 → 128 filtri)
- Flatten + MLP classificatore con Dropout
- Loss: categorical cross-entropy
"""

from tensorflow.keras import layers, Model
from src.config import IMG_SIZE


# ──────────────────────────────────────────────────────────────────────────────
# Blocco residuale custom
# ──────────────────────────────────────────────────────────────────────────────

def residual_block(x, filters, kernel_size=3):
    # Blocco residuale: Conv→BN→ReLU→Conv→BN + skip connection → ReLU.
    # Se i filtri cambiano, la skip connection usa una conv 1x1 per adattare.

    shortcut = x

    # Se il numero di filtri in input è diverso, proiettiamo la skip connection
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Prima convoluzione + batch norm + attivazione
    out = layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)

    # Seconda convoluzione + batch norm (senza attivazione prima della somma)
    out = layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)(out)
    out = layers.BatchNormalization()(out)

    # Somma residuo + input originale, poi attivazione
    out = layers.Add()([out, shortcut])
    out = layers.ReLU()(out)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Costruzione del modello completo
# ──────────────────────────────────────────────────────────────────────────────

def build_model(num_classes, input_shape=None):
    # Costruisce la CNN residuale custom per classificazione.
    # Architettura: Stem → 3 stage con ResBlock → MaxPool → Flatten → MLP

    if input_shape is None:
        input_shape = (IMG_SIZE[1], IMG_SIZE[0], 3)

    inputs = layers.Input(shape=input_shape)

    # ─── Stem: convoluzione iniziale, stride 2 per dimezzare le dimensioni ────
    # Input: 128x128x3 → Output: 64x64x32
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # ─── Stage 1: primo blocco residuale (32 filtri, 64x64) ──────────────────
    x = residual_block(x, 32)

    # ─── Transizione 1→2: downsample con stride 2, aumento filtri a 64 ───────
    # Output: 32x32x64
    x = layers.Conv2D(64, 3, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # ─── Stage 2: secondo blocco residuale (64 filtri, 32x32) ────────────────
    x = residual_block(x, 64)

    # ─── Transizione 2→3: downsample con stride 2, aumento filtri a 128 ──────
    # Output: 16x16x128
    x = layers.Conv2D(128, 3, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # ─── Stage 3: terzo blocco residuale (128 filtri, 16x16) ─────────────────
    x = residual_block(x, 128)

    # ─── Riduzione spaziale con MaxPooling (NO Average Pooling!) ─────────────
    # 16x16 → 4x4 con pool 4x4
    x = layers.MaxPooling2D(pool_size=(4, 4))(x)

    # ─── Flatten: appiattisce il feature map in un vettore 1D ────────────────
    # 4x4x128 = 2048
    x = layers.Flatten()(x)

    # ─── MLP classificatore: 1 hidden layer + output ─────────────────────────
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)

    # Layer di output con softmax per le probabilità di classe
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name='custom_resnet')
    return model
