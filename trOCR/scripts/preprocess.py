#preprocess.py

import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0" # Disabilita backend problematici su Windows
import cv2
import numpy as np
from PIL import Image


#pulizia aggressiva
def clean_image(image_path):
    # Carica l'immagine in scala di grigi
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 1. Ridimensionamento: TrOCR preferisce altezze standard (es. 384px)
    # Manteniamo il rapporto d'aspetto per non deformare le lettere
    height, width = img.shape
    new_height = 384
    new_width = int((new_height / height) * width)
    img = cv2.resize(img, (new_width, new_height))

    # 2. Binarizzazione: Trasforma tutto in bianco e nero netto
    # Utile per eliminare ombre della carta o macchie
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)

    # 3. Denoising: Rimuove i "puntini" di disturbo
    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

    # Converte da OpenCV (array) a PIL Image (richiesto dal Processor di Hugging Face)
    return Image.fromarray(img).convert("RGB")


#pulizia soft
def clean_cell_for_ocr(cv2_img):
    # Converti in scala di grigi se non lo è
    if len(cv2_img.shape) == 3:
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2_img

    # Invece di adaptive, usiamo un OTSU semplice che mantiene meglio il tratto a mano
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Riporta a PIL per TrOCR
    return Image.fromarray(thresh).convert("RGB")