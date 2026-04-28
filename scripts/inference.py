#inference.py


import torch
from jiwer import cer, wer
import matplotlib.pyplot as plt


def run_prediction(model, processor, image, device):
    """Esegue l'OCR su una singola immagine pulita."""
    # Preparazione dei pixel values
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    generated_ids = model.generate(
        pixel_values,
        max_length=64,
        num_beams=5,  # Esplora 5 percorsi possibili
        early_stopping=True,
        no_repeat_ngram_size=2
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def calculate_metrics(predicted_text, ground_truth):
    """Calcola CER e WER tra la predizione e il testo reale."""
    error_c = cer(ground_truth, predicted_text)
    error_w = wer(ground_truth, predicted_text)
    return {"cer": error_c, "wer": error_w}


def plot_results(metrics):
    """Genera un grafico a barre degli errori."""
    labels = ['CER (Caratteri)', 'WER (Parole)']
    values = [metrics['cer'], metrics['wer']]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values, color=['#3498db', '#e74c3c'])
    plt.ylim(0, max(values) + 0.1 if max(values) < 1 else 1.2)
    plt.ylabel('Tasso di Errore')
    plt.title('Performance OCR TrOCR')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.2%}", ha='center', fontweight='bold')

    plt.show()


