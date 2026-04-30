# prepare_qwen_data.py


import pandas as pd
import json
import os

def convert_to_qwen_format(csv_path, output_jsonl):
    df = pd.read_csv(csv_path, sep=";")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            # Creiamo una struttura a "chat"
            entry = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": row['file_name']},
                            {"type": "text", "text": "Trascrivi esattamente il testo contenuto in questa immagine."}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": str(row['text'])}
                        ]
                    }
                ]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    convert_to_qwen_format("data/metadata.csv", "data/dataset_qwen.jsonl")
    print("Dataset convertito in data/dataset_qwen.jsonl")