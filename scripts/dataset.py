# dataset.py

import torch
from torch.utils.data import Dataset
import cv2
from scripts.preprocess import clean_cell_for_ocr


class OCRDataset(Dataset):
    def __init__(self, df, processor, max_target_length=128):
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]

        # CORREZIONE: Carica l'immagine come array OpenCV prima di pulirla
        cv2_img = cv2.imread(file_name)
        if cv2_img is None:
            raise FileNotFoundError(f"Impossibile caricare: {file_name}")

        image = clean_cell_for_ocr(cv2_img)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True
        ).input_ids

        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}