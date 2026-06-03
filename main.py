"""
Assignment 3 - Visione Artificiale
Classificazione di Immagini Aeree tramite CNN Residuale Custom

Script principale: esegue l'intera pipeline
  1. Caricamento e split dei dati
  2. Pre-addestramento su AID (Strategia 1, fase 1)
  3. Fine-tuning su UC Merced (Strategia 1, fase 2)
  4. Addestramento da zero su UC Merced (Strategia 2)
  5. Model selection con F1-score macro
  6. Valutazione finale sul test set
  7. Generazione plot e risultati qualitativi

Usa: python main.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
from src.data_loader import load_ucmerced_split, load_aid_split
from src.train import pretrain_on_aid, finetune_on_ucmerced, train_from_scratch
from src.evaluate import model_selection, evaluate_on_test, plot_training_curves


def main():
    start = time.time()

    print("█" * 60)
    print("  ASSIGNMENT 3 — CNN Residuale Custom")
    print("  Classificazione Immagini Aeree")
    print("█" * 60)

    # ─── 1. Verifica dati ────────────────────────────────────────────────────
    print("\n[1/6] Verifica dataset e split...")
    load_ucmerced_split()
    load_aid_split()

    # ─── 2. Pre-addestramento su AID ─────────────────────────────────────────
    print("\n[2/6] Pre-addestramento su AID...")
    pretrain_on_aid()

    # ─── 3. Fine-tuning su UC Merced ─────────────────────────────────────────
    print("\n[3/6] Fine-tuning su UC Merced...")
    finetune_on_ucmerced()

    # ─── 4. Training da zero su UC Merced ────────────────────────────────────
    print("\n[4/6] Training da zero su UC Merced...")
    train_from_scratch()

    # ─── 5. Model Selection ──────────────────────────────────────────────────
    print("\n[5/6] Model selection (F1-score macro)...")
    best_model, best_name, f1_ft, f1_sc = model_selection()

    # ─── 6. Valutazione finale + plot ────────────────────────────────────────
    print("\n[6/6] Valutazione sul test set e generazione plot...")
    acc, prec, rec, f1, cm = evaluate_on_test(best_model, best_name)
    plot_training_curves()

    # ─── Riepilogo finale ────────────────────────────────────────────────────
    total_min = (time.time() - start) / 60
    print("\n" + "█" * 60)
    print("  PIPELINE COMPLETATA")
    print("█" * 60)
    print(f"  Tempo totale:      {total_min:.1f} minuti")
    print(f"  Modello migliore:  {best_name}")
    print(f"  F1 val (fine-tuned):    {f1_ft:.4f}")
    print(f"  F1 val (from-scratch):  {f1_sc:.4f}")
    print(f"  Test Accuracy:     {acc:.4f}")
    print(f"  Test F1 macro:     {f1:.4f}")
    print()
    print("  Per classificare nuove immagini:")
    print("    python inference.py <percorso_immagine>")
    print()


if __name__ == '__main__':
    main()
