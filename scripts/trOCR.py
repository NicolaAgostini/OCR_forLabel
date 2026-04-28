#trOCR.py

import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
import sys
from sklearn.model_selection import train_test_split


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"  # Fondamentale per Blackwell
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd



def trOCR():

    import torch
    from PIL import Image
    import cv2

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel


    # IMPORT LOCALE DEI TUOI SCRIPT
    from scripts.dataset import OCRDataset
    from scripts.train_engine import run_fine_tuning
    from scripts.inference import run_prediction
    from scripts.table_extractor import extract_cell_crops


    # ... resto del codice (caricamento dati, ecc.) ...
    print("Avvio logica applicativa...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("running on device: ", device)
    #model_name = "microsoft/trocr-base-handwritten"
    model_name = "microsoft/trocr-large-handwritten"
    processor = TrOCRProcessor.from_pretrained(model_name)

    mode = input("Scegli modalità (train/inferenza): ").strip().lower()

    if mode == "train":
        # 1. Caricamento metadati (CSV con colonne 'file_name' e 'text')
        df = pd.read_csv("data/metadata.csv", sep=";")
        # security check
        if 'file_name' not in df.columns:
            print(f"ERRORE: Colonne trovate: {list(df.columns)}")
            print("Verifica che il separatore nel CSV sia ';' e che l'intestazione sia corretta.")
            return
        train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)
        train_df.reset_index(drop=True, inplace=True)
        eval_df.reset_index(drop=True, inplace=True)

        # 2. Creazione oggetti Dataset
        train_dataset = OCRDataset(train_df, processor)
        eval_dataset = OCRDataset(eval_df, processor)

        # 3. Caricamento modello e avvio training
        model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)

        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size

        model.to(device)

        print("Inizio fase di Fine-Tuning...")
        run_fine_tuning(model, processor, train_dataset, eval_dataset)


    elif mode == "inferenza":

        path_modello = "./trocr-fine-tuned-ita"

        if os.path.exists(path_modello):
            print(f"Caricamento modello locale: {path_modello}")
            model = VisionEncoderDecoderModel.from_pretrained(path_modello).to(device)
            processor = TrOCRProcessor.from_pretrained(path_modello)
        else:
            model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
            processor = TrOCRProcessor.from_pretrained(model_name)
        img_path = input("Path immagine tabella: ").strip()

        if os.path.exists(img_path):
            print("--- Estrazione celle dalla tabella ---")
            # Chiamata alla funzione in table_extractor.py
            cells = extract_cell_crops(img_path)
            print(f"Trovate {len(cells)} celle. Inizio riconoscimento...")
            risultati = []

            for i, cell_img in enumerate(cells):
                # Conversione da formato OpenCV (BGR) a PIL (RGB) richiesto dal modello
                cell_pil = Image.fromarray(cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB))
                # Esecuzione OCR sulla singola cella
                text = run_prediction(model, processor, cell_pil, device)
                if text.strip():
                    print(f"Cella {i + 1}: {text}")
                    risultati.append(text)

            print("\n--- RISULTATO FINALE ---")
            print(" | ".join(risultati))

        else:

            print("File non trovato.")







