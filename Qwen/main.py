# main.py

from unsloth import FastVisionModel, is_bf16_supported
import torch
import sys

# DOPO puoi importare trl o transformers
from trl import SFTTrainer
from transformers import TrainingArguments
import os
from scripts.inference import run_qwen_inference
from scripts.train_qwen import run_qwen_train

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Forza l'uso di Unsloth come prima cosa
try:
    from unsloth import FastVisionModel
except ImportError:
    print("Errore: Unsloth non installato correttamente.")


def main():
    mode = input("Scegli modalità (train/inferenza/full): ").strip().lower()

    fine_tuned_path = "qwen_fine_tuned"
    base_model_path = "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"

    # Determiniamo quale modello usare per l'inferenza
    if os.path.exists(fine_tuned_path):
        model_to_use = fine_tuned_path
        status_msg = f"🚀 Utilizzo MODELLO FINE-TUNED: {fine_tuned_path}"
    else:
        model_to_use = base_model_path
        status_msg = f"💡 Utilizzo MODELLO BASE (Nessun fine-tuning trovato): {base_model_path}"
    if mode == "train":
        run_qwen_train()

    elif mode == "inferenza":

        print("\n" + "=" * 50)
        print(status_msg)
        print("=" * 50 + "\n")

        path_input = input("Inserisci il path del file o della cartella immagini: ").strip()

        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

        if os.path.isdir(path_input):
            files = [os.path.join(path_input, f) for f in os.listdir(path_input)
                     if f.lower().endswith(valid_extensions)]
            print(f"Trovate {len(files)} immagini da processare.")
        elif os.path.isfile(path_input) and path_input.lower().endswith(valid_extensions):
            files = [path_input]
        else:
            print("❌ Errore: Percorso non valido o nessuna immagine trovata.")
            return

        for img_path in files:
            run_qwen_inference(model_to_use, img_path)


if __name__ == "__main__":
    main()