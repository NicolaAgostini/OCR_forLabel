# inference.py

import torch
from unsloth import FastVisionModel
from PIL import Image
import csv
import os


def run_qwen_inference(model_path, image_path):
    """Esegue l'inferenza su una tabella intera."""
    print(f"\n--- INFERENZA SU: {image_path} ---")

    # Caricamento ottimizzato per Blackwell (sm_120) e 8GB VRAM
    # Se model_path è "unsloth/Qwen2.5-VL-7B...", prova a usare il 3B se crasha
    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=True,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    FastVisionModel.for_inference(model)

    image = Image.open(image_path).convert("RGB")

    # 1. Definizione dei messaggi
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},  # Passiamo l'oggetto PIL direttamente
                {"type": "text",
                 "text": "Sei un agente OCR: trascrivi questa tabella in formato Markdown. Mantieni tutte le colonne e interpreta le virgolette (\") come ripetizione del valore sopra."}
            ]
        }
    ]

    # 2. Applichiamo il template (tokenize=False è fondamentale qui)
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 3. PREPARAZIONE INPUT (Corretta per evitare il ValueError)
    # Non passare l'immagine dentro il tokenizer() se usi il testo già templatizzato.
    # Usiamo il processor in modo esplicito se disponibile, o configuriamo il tokenizer.
    inputs = tokenizer(
        text=[input_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # 4. GENERAZIONE
    # Per tabelle grandi, aumentiamo max_new_tokens e usiamo min_pixels/max_pixels se necessario
    outputs = model.generate(
        **inputs,
        max_new_tokens=1536,  # Aumentato per tabelle lunghe
        use_cache=True,
        temperature=0.1,  # Bassa per maggiore precisione nell'OCR
        top_p=0.9,
    )

    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Pulizia dell'output (rimuove il prompt dell'utente)
    if "assistant" in prediction:
        final_output = prediction.split("assistant")[-1].strip()
    else:
        final_output = prediction

    print("\nRISULTATO ESTRATTO:\n")
    print(final_output)

    # --- LOGICA SALVATAGGIO CSV ---
    # Genera il nome file .csv basandosi sull'immagine originale
    csv_filename = os.path.splitext(image_path)[0] + ".csv"

    lines = final_output.strip().split('\n')
    table_data = []

    for line in lines:
        # Filtriamo le righe che contengono il separatore Markdown '|' escludendo la riga di intestazione '---'
        if '|' in line and not all(c in '|- ' for c in line.strip()):
            # Pulizia delle celle
            row = [cell.strip() for cell in line.split('|') if cell.strip()]
            if row:
                table_data.append(row)

    if table_data:
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
            # Uso ';' come delimitatore per compatibilità Excel automatica
            writer = csv.writer(f, delimiter=';')
            writer.writerows(table_data)
        print(f"✅ Risultato salvato: {csv_filename}")
    else:
        print(f"⚠️ Nessuna tabella rilevata in {image_path}")

    # Pulizia memoria VRAM per il prossimo file
    del model, inputs, outputs
    torch.cuda.empty_cache()