"""
Modulo: Valutazione e metriche
- Model selection: confronto F1-score (macro) sul validation set
- Valutazione finale sul test set: Accuracy, Precision, Recall, F1, Confusion Matrix
- Risultati qualitativi: immagini classificate correttamente e erroneamente
- Plot curve di training/validation (loss e accuracy)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non interattivo per salvare i plot
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from tensorflow import keras
from src.config import (
    FINETUNED_MODEL, SCRATCH_MODEL, BEST_MODEL, PROCESSED_DIR,
    NUM_CLASSES_UCMERCED, UCMERCED_CLASSES
)
from src.data_loader import load_ucmerced_split, load_images_from_paths


# ──────────────────────────────────────────────────────────────────────────────
# Model Selection: confronto F1-score sul validation set
# ──────────────────────────────────────────────────────────────────────────────

def model_selection():
    # Confronta i due modelli sul validation set usando F1-score macro-average.
    # Seleziona il migliore e lo salva come best_model.

    print("\n" + "=" * 60)
    print("MODEL SELECTION (F1-score macro sul validation set)")
    print("=" * 60)

    # Carica lo split salvato
    tr_p, tr_l, v_p, v_l, te_p, te_l, class_names = load_ucmerced_split()

    print("\nCaricamento immagini validation...")
    X_val, y_val = load_images_from_paths(v_p, v_l, class_names)

    # Carica i due modelli addestrati
    print(f"Caricamento modello fine-tuned: {FINETUNED_MODEL}")
    model_ft = keras.models.load_model(str(FINETUNED_MODEL))

    print(f"Caricamento modello from-scratch: {SCRATCH_MODEL}")
    model_sc = keras.models.load_model(str(SCRATCH_MODEL))

    # Predizioni su validation set
    y_pred_ft = np.argmax(model_ft.predict(X_val, verbose=0), axis=1)
    y_pred_sc = np.argmax(model_sc.predict(X_val, verbose=0), axis=1)

    # F1-score macro-average (unica metrica per la selezione)
    f1_ft = f1_score(y_val, y_pred_ft, average='macro')
    f1_sc = f1_score(y_val, y_pred_sc, average='macro')

    print(f"\n  Fine-tuned  — F1 macro: {f1_ft:.4f}")
    print(f"  From-scratch — F1 macro: {f1_sc:.4f}")

    # Seleziona il modello migliore
    if f1_ft >= f1_sc:
        best_name = "Fine-tuned (Strategia 1)"
        best_model = model_ft
        best_path = FINETUNED_MODEL
    else:
        best_name = "From-scratch (Strategia 2)"
        best_model = model_sc
        best_path = SCRATCH_MODEL

    print(f"\n  ★ Modello selezionato: {best_name}")

    # Salva una copia come best_model
    best_model.save(str(BEST_MODEL))
    print(f"  Salvato come: {BEST_MODEL}")

    return best_model, best_name, f1_ft, f1_sc


# ──────────────────────────────────────────────────────────────────────────────
# Valutazione finale sul test set
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_on_test(model, model_name="best_model"):
    # Calcola tutte le metriche sul test set: Accuracy, Precision, Recall,
    # F1-score (macro-average) e Confusion Matrix.

    print("\n" + "=" * 60)
    print(f"VALUTAZIONE SUL TEST SET — {model_name}")
    print("=" * 60)

    # Carica il test set
    tr_p, tr_l, v_p, v_l, te_p, te_l, class_names = load_ucmerced_split()

    print("\nCaricamento immagini test...")
    X_test, y_test = load_images_from_paths(te_p, te_l, class_names)

    # Predizioni
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calcolo metriche
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    # Stampa risultati
    print(f"\n  Accuracy:        {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Precision macro: {prec:.4f}")
    print(f"  Recall macro:    {rec:.4f}")
    print(f"  F1-score macro:  {f1:.4f}")

    # Classification report per classe
    print(f"\n  Report per classe:")
    report = classification_report(y_test, y_pred, target_names=class_names, digits=3)
    print(report)

    # Salva la confusion matrix come immagine
    plot_confusion_matrix(cm, class_names)

    # Risultati qualitativi: immagini corrette e errate
    plot_qualitative_results(X_test, y_test, y_pred, y_pred_probs, class_names)

    return acc, prec, rec, f1, cm


# ──────────────────────────────────────────────────────────────────────────────
# Plot Confusion Matrix
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, class_names):
    # Salva la confusion matrix come heatmap in data/processed/

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='Label reale',
        xlabel='Label predetta',
        title='Confusion Matrix — Test Set'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    # Annota ogni cella con il conteggio
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=6)

    plt.tight_layout()
    path = str(PROCESSED_DIR / "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix salvata in: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Risultati qualitativi: esempi corretti e errati
# ──────────────────────────────────────────────────────────────────────────────

def plot_qualitative_results(X_test, y_test, y_pred, y_pred_probs, class_names):
    # Mostra 5 immagini classificate correttamente e 5 erroneamente,
    # con la distribuzione delle probabilità top-5.

    correct = np.where(y_test == y_pred)[0]
    wrong = np.where(y_test != y_pred)[0]

    # Seleziona 5 esempi per tipo (o meno se non ce ne sono abbastanza)
    n_examples = min(5, len(correct), len(wrong))
    correct_idx = np.random.choice(correct, n_examples, replace=False)
    wrong_idx = np.random.choice(wrong, n_examples, replace=False)

    for tag, indices in [("corrette", correct_idx), ("errate", wrong_idx)]:
        fig, axes = plt.subplots(1, n_examples, figsize=(4 * n_examples, 5))
        if n_examples == 1:
            axes = [axes]

        for ax, idx in zip(axes, indices):
            ax.imshow(X_test[idx])
            ax.axis('off')

            true_label = class_names[y_test[idx]]
            pred_label = class_names[y_pred[idx]]
            prob = y_pred_probs[idx][y_pred[idx]]

            # Titolo: label vera e predetta con probabilità
            color = 'green' if tag == "corrette" else 'red'
            ax.set_title(
                f"Vera: {true_label}\nPred: {pred_label} ({prob:.1%})",
                fontsize=9, color=color
            )

        plt.suptitle(f"Classificazioni {tag}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = str(PROCESSED_DIR / f"qualitative_{tag}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Risultati qualitativi ({tag}) salvati in: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot curve di training (loss e accuracy)
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves():
    # Carica le history salvate e genera i plot loss/accuracy
    # per pre-training, fine-tuning e from-scratch.

    print("\n" + "=" * 60)
    print("GENERAZIONE PLOT DI TRAINING")
    print("=" * 60)

    history_files = {
        "Pre-training AID": "history_pretrain_aid.json",
        "Fine-tuning UC Merced": "history_finetune.json",
        "From-scratch UC Merced": "history_scratch.json"
    }

    for name, filename in history_files.items():
        filepath = PROCESSED_DIR / filename
        if not filepath.exists():
            print(f"  SKIP: {filename} non trovato")
            continue

        with open(str(filepath), 'r') as f:
            hist = json.load(f)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(hist['loss']) + 1)

        # Plot Loss
        ax1.plot(epochs, hist['loss'], 'b-', label='Training')
        ax1.plot(epochs, hist['val_loss'], 'r-', label='Validation')
        ax1.set_title(f'{name} — Loss')
        ax1.set_xlabel('Epoca')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot Accuracy
        ax2.plot(epochs, hist['accuracy'], 'b-', label='Training')
        ax2.plot(epochs, hist['val_accuracy'], 'r-', label='Validation')
        ax2.set_title(f'{name} — Accuracy')
        ax2.set_xlabel('Epoca')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_name = filename.replace('.json', '.png')
        path = str(PROCESSED_DIR / plot_name)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  {name}: {path}")
