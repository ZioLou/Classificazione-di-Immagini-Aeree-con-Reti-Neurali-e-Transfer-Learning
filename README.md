# Classificazione di Immagini Aeree tramite CNN Residuale e Transfer Learning 🛰️

Progetto per l'**Assignment 3** del corso di Visione Artificiale. 
L'obiettivo è la classificazione del dataset di immagini aeree **UC Merced (21 classi)** confrontando l'efficacia dell'addestramento *from-scratch* rispetto a tecniche di *Transfer Learning* (pre-addestramento sul dataset affine **AID** e successivo fine-tuning).

Il sistema implementa una **Convolutional Neural Network (CNN) Residuale Custom** sviluppata in TensorFlow/Keras.

## 🗂️ Struttura del Progetto

```text
progetto_assignment3/
├── main.py              # Script principale: esegue l'intera pipeline di training e test
├── inference.py         # Script per testare il modello su immagini singole
├── requirements.txt     # Dipendenze del progetto
├── data/
│   ├── raw/             # [NON INCLUSO IN GIT] Dataset originali (AID, UCMerced)
│   └── processed/       # Risultati, metriche e split stratificati (splits.npz)
├── src/
│   ├── config.py        # Costanti, percorsi e iperparametri (LR, epoche, batch)
│   ├── data_loader.py   # Caricamento, resize, normalizzazione OpenCV e split (70/15/15)
│   ├── model.py         # Architettura della CNN Residuale
│   ├── train.py         # Funzioni di training (pretrain, finetune, scratch)
│   └── evaluate.py      # Metriche (F1, Accuracy), matrice di confusione e plot
└── models/              # [NON INCLUSO IN GIT] Modelli salvati (.keras)
```

## ⚙️ Requisiti e Setup (Per il Docente)

I file `.keras` dei modelli addestrati e i dataset originali **non sono inclusi** nella repository a causa delle loro dimensioni.

### 1. Preparazione dell'Ambiente
È consigliato l'utilizzo di un ambiente virtuale (Python 3.9+).
```bash
python3 -m venv venv
source venv/bin/activate  # (Su Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

### 2. Download dei Dataset
Scaricare i dataset e posizionarli esattamente all'interno della cartella `data/raw/` in modo che la struttura risulti:
*   `data/raw/AID/` (contenente le 30 cartelle delle classi)
*   `data/raw/UCMerced_LandUse/Images/` (contenente le 21 cartelle delle classi)

*(I percorsi esatti possono comunque essere modificati all'interno di `src/config.py` in caso di necessità).*

## 🚀 Utilizzo

### Riprodurre l'Addestramento Completo
Per eseguire l'intera pipeline sperimentale (pre-training AID -> fine-tuning UC Merced -> training from scratch UC Merced -> confronto finale e plotting):
```bash
python main.py
```
Tutti i grafici di learning (accuracy/loss), la matrice di confusione e i confronti verranno salvati in automatico nella cartella `data/processed/`.

### Inferenza su Singola Immagine
Se è già presente un modello addestrato nella cartella `models/`, è possibile testarlo su una nuova immagine:
```bash
python inference.py <percorso/alla/tua/immagine.jpg>
```
