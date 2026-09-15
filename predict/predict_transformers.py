import json
import os
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

ID2LABEL = {0: "O", 1: "B", 2: "I"}

def predict_single_task(model, tokenizer, words, device, max_len=128):
    """
    Predict labels for a single word list, ensuring the output length strictly matches the input words
    """
    inputs = tokenizer(
        words,
        is_split_into_words=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # shape: [1, seq_len, 3]
        predictions = torch.argmax(logits, dim=-1).squeeze(0).tolist()

    word_ids = inputs.word_ids(batch_index=0)
    
    # Align subword dimensions back to original words dimensions (first token strategy)
    pred_labels = []
    prev_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != prev_word_idx:
            pred_labels.append(ID2LABEL.get(predictions[idx], "O"))
            prev_word_idx = word_idx

    # In extreme cases where sentences exceed max_len and are truncated, fill missing endings with "O"
    if len(pred_labels) < len(words):
        pred_labels.extend(["O"] * (len(words) - len(pred_labels)))
    elif len(pred_labels) > len(words):
        pred_labels = pred_labels[:len(words)]

    return pred_labels

def run_inference(test_empty_path, checkpoint_dir, output_pred_path, device):
    skill_dir = os.path.join(checkpoint_dir, "skill")
    know_dir = os.path.join(checkpoint_dir, "knowledge")

    if not os.path.exists(skill_dir) or not os.path.exists(know_dir):
        print(f"[Skip] Complete weight directory not found: {checkpoint_dir}")
        return

    print(f"\nLoading weights: {checkpoint_dir}")
    # Load Skill model
    tokenizer_skill = AutoTokenizer.from_pretrained(skill_dir)
    model_skill = AutoModelForTokenClassification.from_pretrained(skill_dir).to(device)
    model_skill.eval()

    # Load Knowledge model
    tokenizer_know = AutoTokenizer.from_pretrained(know_dir)
    model_know = AutoModelForTokenClassification.from_pretrained(know_dir).to(device)
    model_know.eval()

    total_count = 0
    with open(test_empty_path, "r", encoding="utf-8") as f_in, open(output_pred_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            tokens = sample["tokens"]

            pred_skills = predict_single_task(model_skill, tokenizer_skill, tokens, device)
            pred_knows = predict_single_task(model_know, tokenizer_know, tokens, device)

            # Assemble prediction row
            result_item = {
                "idx": sample.get("idx"),
                "tokens": tokens,
                "tags_skill": pred_skills,
                "tags_knowledge": pred_knows,
                "source": sample.get("source", "")
            }

            f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            total_count += 1

    print(f"[✓] Inference complete! Prediction file written to: {output_pred_path} (Total {total_count} rows)")

    # Release GPU memory
    del model_skill, tokenizer_skill, model_know, tokenizer_know
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # ==================== Centralized Path and Model Dictionary Configuration ====================
    TEST_EMPTY_FILE = "/home/tu/tu_tu/tu_zxois72/26ss/NER_Job/Data/raw/test_empty.json"
    OUTPUT_DATA_DIR = "/home/tu/tu_tu/tu_zxois72/26ss/NER_Job/Data/predictions"
    CHECKPOINTS_BASE = "./checkpoints"

    # Model suffix mapping
    MODELS_TO_PREDICT = {
        "bert_base_cased": "checkpoints_bert_base_cased",
        "jobbert_base_cased": "checkpoints_jobbert_base_cased"
    }
    # ==============================================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current inference execution device: {device}")

    for suffix, ckpt_name in MODELS_TO_PREDICT.items():
        ckpt_path = os.path.join(CHECKPOINTS_BASE, ckpt_name)
        out_file = os.path.join(OUTPUT_DATA_DIR, f"pred_{suffix}.json")
        
        run_inference(
            test_empty_path=TEST_EMPTY_FILE,
            checkpoint_dir=ckpt_path,
            output_pred_path=out_file,
            device=device
        )