"""
Modulo: Inferenza
- Carica il modello migliore (selezionato con F1-score macro)
- Classifica nuove immagini aeree da riga di comando
- Mostra le top-3 predizioni con le relative probabilità

Usa: python inference.py <percorso_immagine> [percorso_immagine_2 ...]
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow import keras
from src.data_loader import load_image
from src.config import BEST_MODEL, UCMERCED_CLASSES, IMG_SIZE


def load_best_model():
    # Carica il modello migliore salvato durante la model selection
    if not BEST_MODEL.exists():
        print(f"ERRORE: modello non trovato in {BEST_MODEL}")
        print("Eseguire prima il training e la model selection.")
        sys.exit(1)

    print(f"Caricamento modello: {BEST_MODEL}")
    model = keras.models.load_model(str(BEST_MODEL))
    return model


def predict_image(model, image_path):
    # Classifica una singola immagine e restituisce le top-3 predizioni

    # Carica e preprocessa l'immagine (OpenCV → RGB → resize → normalizza)
    img = load_image(image_path)

    # Aggiunge la dimensione batch: (H, W, 3) → (1, H, W, 3)
    img_batch = np.expand_dims(img, axis=0)

    # Predizione
    probs = model.predict(img_batch, verbose=0)[0]

    # Ordina per probabilità decrescente e prende le top-3
    top3_idx = np.argsort(probs)[::-1][:3]
    results = [(UCMERCED_CLASSES[i], probs[i]) for i in top3_idx]

    return results, img


def main():
    if len(sys.argv) < 2:
        print("Uso: python inference.py <percorso_immagine> [...]")
        print("Esempio: python inference.py data/raw/UCMerced_LandUse/Images/beach/beach00.tif")
        sys.exit(1)

    model = load_best_model()

    # Classifica ogni immagine passata come argomento
    for image_path in sys.argv[1:]:
        if not os.path.exists(image_path):
            print(f"\nERRORE: file non trovato: {image_path}")
            continue

        results, _ = predict_image(model, image_path)

        print(f"\n{'─'*50}")
        print(f"Immagine: {image_path}")
        print(f"{'─'*50}")
        print(f"  Predizione: {results[0][0]} ({results[0][1]:.1%})")
        print(f"  Top-3:")
        for rank, (label, prob) in enumerate(results, 1):
            bar = '█' * int(prob * 30)
            print(f"    {rank}. {label:20s} {prob:6.1%} {bar}")


if __name__ == '__main__':
    main()
