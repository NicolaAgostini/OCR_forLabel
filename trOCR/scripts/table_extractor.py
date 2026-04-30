# table_extractor.py
import cv2
import numpy as np
import os


def extract_cell_crops(image_path, output_folder="data/extracted"):
    # Crea la cartella se non esiste
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # Pulisce la cartella da vecchi ritagli
        for f in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, f))

    img = cv2.imread(image_path)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    table_mask = cv2.addWeighted(detect_horizontal, 0.5, detect_vertical, 0.5, 0.0)
    table_mask = cv2.threshold(table_mask, 0, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    cells = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 40 and h > 20:
            cells.append((x, y, w, h))

    # Ordinamento top-to-bottom, left-to-right
    cells = sorted(cells, key=lambda b: (b[1], b[0]))

    crops = []
    for i, (x, y, w, h) in enumerate(cells):
        padding = 2
        crop = img[y + padding:y + h - padding, x + padding:x + w - padding]

        if crop.size > 0:
            bordo = 10
            crop_with_border = cv2.copyMakeBorder(
                crop,
                bordo, bordo, bordo, bordo,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255]  # Bianco
            )

            cell_filename = os.path.join(output_folder, f"cell_{i + 1}.png")
            cv2.imwrite(cell_filename, crop_with_border)
            crops.append(crop_with_border)

    return crops