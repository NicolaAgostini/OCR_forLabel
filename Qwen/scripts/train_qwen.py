# train_qwen.py



import os
import torch
from unsloth import FastVisionModel, is_bf16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Ottimizzazione VRAM per Blackwell 8GB
os.environ["UNSLOTH_MAX_IMAGE_SIZE"] = "384"


def run_qwen_train():
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
        load_in_4bit=True,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        # 2. Riduciamo il carico della cache
        use_gradient_checkpointing="unsloth",
    )



    model = FastVisionModel.get_peft_model(
        model,
        r=16,  #  rango
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        random_state=3407,
    )

    dataset = load_dataset("json", data_files={"train": "data/dataset_qwen.jsonl"}, split="train")

    def apply_template(sample):
        text = tokenizer.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}

    dataset = dataset.map(apply_template, batched=False)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            #max_steps=30,  # test breve
            num_train_epochs=3,
            learning_rate=2e-4,
            bf16=True,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,
            gradient_checkpointing=True,
        ),
    )

    trainer.train()

    # 4. Salvataggio
    model.save_pretrained("qwen_fine_tuned")
    tokenizer.save_pretrained("qwen_fine_tuned")
    print("\nTraining completato! Modello salvato in 'qwen_fine_tuned'")