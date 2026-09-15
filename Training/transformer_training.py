import argparse
import json
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

LABEL2ID = {"O": 0, "B": 1, "I": 2}
ID2LABEL = {0: "O", 1: "B", 2: "I"}

class SkillSpanDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128, task="skill"):
        self.data = []
        tag_key = f"tags_{task}"
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                self.data.append((item["tokens"], item[tag_key]))
                
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, tags = self.data[idx]
        tokenized_inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        word_ids = tokenized_inputs.word_ids(batch_index=0)
        labels = []
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            else:
                tag = tags[word_idx]
                labels.append(LABEL2ID[tag])
                
        item = {key: val.squeeze(0) for key, val in tokenized_inputs.items()}
        item["labels"] = torch.tensor(labels, dtype=torch.long)
        return item

def train_single_task(model_name, task, train_path, epochs, batch_size, lr, device, output_dir):
    print(f"\n  [Task: {task.upper()}] Fine-tuning...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = SkillSpanDataset(train_path, tokenizer, max_len=128, task=task)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1), 
        num_training_steps=total_steps
    )

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")

    # Save final weights
    task_output_dir = os.path.join(output_dir, task)
    os.makedirs(task_output_dir, exist_ok=True)
    model.save_pretrained(task_output_dir)
    tokenizer.save_pretrained(task_output_dir)
    print(f"  [✓] {task.upper()} weights successfully saved to: {task_output_dir}")

    # Do not execute any manual del statements, local variables are automatically reclaimed when the function exits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
   
    TRAIN_PATH = "/home/tu/tu_tu/tu_zxois72/26ss/NER_Job/Data/raw/train_merged.json"
    EPOCHS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-5
    OUTPUT_BASE_DIR = "./checkpoints"

    # Model dictionary: "Model path or HF name": "Save directory suffix"
    MODELS_TO_TRAIN = {
        "bert-base-cased": "bert_base_cased",
        "jjzha/jobbert-base-cased": "jobbert_base_cased"
    }
    # ==============================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current execution device: {device}")

    for model_name, suffix in MODELS_TO_TRAIN.items():
        print(f"\n{'='*60}")
        print(f"Starting training model: {model_name}")
        print(f"{'='*60}")

        save_dir = os.path.join(OUTPUT_BASE_DIR, f"checkpoints_{suffix}")
        os.makedirs(save_dir, exist_ok=True)

        train_single_task(
            model_name=model_name,
            task="skill",
            train_path=TRAIN_PATH,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            device=device,
            output_dir=save_dir
        )
        
        train_single_task(
            model_name=model_name,
            task="knowledge",
            train_path=TRAIN_PATH,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            device=device,
            output_dir=save_dir
        )

        print(f"[✓] Model {model_name} training completed, outputs saved in: {save_dir}")

    print("\nAll model training sessions fully completed!")