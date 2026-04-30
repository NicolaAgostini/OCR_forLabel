# scripts/train_engine.py

import os
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator, TrainerCallback

# Rimuoviamo gli import pesanti dalla cima del file per evitare il crash all'avvio
# cer_metric = evaluate.load("cer") <-- NON FARLO QUI

class VisualizerCallback(TrainerCallback):
    def __init__(self, dataset, output_dir="debug_images"):
        self.dataset = dataset
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0 and state.global_step > 0:
            # Import locale per evitare conflitti all'avvio
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            item = self.dataset[0]
            pixel_values = item["pixel_values"].squeeze(0)
            img_tensor = (pixel_values - pixel_values.min()) / (pixel_values.max() - pixel_values.min())
            img_array = img_tensor.permute(1, 2, 0).cpu().numpy()

            plt.figure(figsize=(10, 2))
            plt.imshow(img_array)
            plt.title(f"Step {state.global_step}")
            plt.axis('off')
            plt.savefig(f"{self.output_dir}/step_{state.global_step}.png")
            plt.close()
            print(f"\n[INFO] Anteprima salvata in {self.output_dir}/step_{state.global_step}.png")

def run_fine_tuning(model, processor, train_dataset, eval_dataset):
    # Carichiamo evaluate solo qui dentro
    import evaluate
    # Forziamo il caricamento della metrica senza usare multiprocessing eccessivo
    cer_metric = evaluate.load("cer")

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        eval_strategy="steps",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        fp16=True, # Ottimale per la tua RTX Blackwell
        output_dir="./trocr-checkpoint",
        logging_steps=10,
        save_steps=500,
        eval_steps=100,
        num_train_epochs=10,
        weight_decay=0.01,
        report_to="none",
        dataloader_num_workers=0 # Fondamentale su Windows per evitare crash di memoria
    )

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
        cer_score = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer_score}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        callbacks=[VisualizerCallback(eval_dataset)] # Riaggiungiamo la callback
    )

    print("--- Inizio Training ---")
    trainer.train()

    model.save_pretrained("./trocr-fine-tuned-ita")
    processor.save_pretrained("./trocr-fine-tuned-ita")
    print("--- Training Completato ---")