import os
import sys
import gc
import json
import torch
import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# Add project root to Python path to import prompts.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Training.prompts import SYSTEM_INSTRUCTION

# ================= Global Path Configurations and Hyperparameters =================

MODEL_MAP = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen3-8B",
}

TRAIN_DATA_PATH = "/home/tu/tu_tu/tu_zxois72/26ss/NER_Job/Data/raw/train_entities.jsonl"
CHECKPOINTS_BASE_DIR = "/home/tu/tu_tu/tu_zxois72/26ss/NER_Job/checkpoints"

NUM_TRAIN_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# =================================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Flexible Full-Shot LoRA Training Script")
    parser.add_argument(
        "--models",
        nargs="+",  
        default=["llama", "qwen"],  
        choices=list(MODEL_MAP.keys()),
        help=f"Select models to train from: {list(MODEL_MAP.keys())}"
    )
    return parser.parse_args()


def format_prompts_on_the_fly(batch):
    """Format pure sentence and entities into Chat messages dynamically."""
    all_messages = []
    for sentence, entities in zip(batch["sentence"], batch["entities"]):
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"Text: {sentence}\nReturn JSON strictly."},
            {"role": "assistant", "content": json.dumps(entities, ensure_ascii=False)},
        ]
        all_messages.append(messages)
    return {"messages": all_messages}


def train_single_model(model_key: str, dataset):
    model_id = MODEL_MAP[model_key.lower()]
    clean_model_name = model_id.rstrip("/").split("/")[-1].lower().replace("-", "_")
    output_checkpoint_dir = os.path.join(CHECKPOINTS_BASE_DIR, f"{clean_model_name}_fullshot")
    os.makedirs(output_checkpoint_dir, exist_ok=True)

    print(f"\n==================================================")
    print(f"[Training Target] Key: {model_key} | ID: {model_id}")
    print(f"[Output Checkpoint] {output_checkpoint_dir}")
    print(f"==================================================")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # 2. Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # 3. Configure LoRA
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Configure Training Arguments
    training_args = SFTConfig(
        output_dir=output_checkpoint_dir,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        optim="adamw_torch",
        report_to="none",
        max_length=MAX_SEQ_LENGTH,  
    )

    # 5. Initialize SFTTrainer 
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer,  
    )

    trainer.train()

    print(f"\n[Saving] Saving LoRA weights to {output_checkpoint_dir}")
    trainer.model.save_pretrained(output_checkpoint_dir)
    tokenizer.save_pretrained(output_checkpoint_dir)

    # 6. Memory Cleanup
    del model
    del trainer
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[Done] Finished fine-tuning for {model_key}.\n")


def main():
    args = parse_args()
    print(f"[Init] Selected models to train: {args.models}")
    print(f"[Init] Loading dataset from: {TRAIN_DATA_PATH}")
    
    raw_dataset = load_dataset("json", data_files=TRAIN_DATA_PATH, split="train")
    dataset = raw_dataset.map(format_prompts_on_the_fly, batched=True)

    for model_key in args.models:
        train_single_model(model_key, dataset)

    print("[Finished] All selected models have been fine-tuned.")


if __name__ == "__main__":
    main()


    