import sys
from pathlib import Path
import ast
import gc
import json
import os
import re
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add project root to Python path to import prompts.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Training.prompts import SYSTEM_INSTRUCTION

# ================= Global Path Configurations and Hyperparameters =================
MODEL_MAP = {
    "qwen": "Qwen/Qwen3-8B",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}

CHECKPOINTS_BASE_DIR = "/home/tu/tu_tu/tu_zxois72/26ss/NER_Job/checkpoints"
TEST_EMPTY_PATH = "/home/tu/tu_tu/tu_zxois72/26ss/NER_Job/Data/raw/test_empty.json"
OUTPUT_DIR = "/home/tu/tu_tu/tu_zxois72/26ss/NER_Job/Data/llm_outputs"

BATCH_SIZE = 32
MAX_NEW_TOKENS = 192
DEVICE_MAP = "auto"
# =================================================================================


def parse_args():
    """Parse command line arguments, supporting model specification via --models"""
    parser = argparse.ArgumentParser(description="Full-Shot Inference Script")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen", "llama"],
        choices=list(MODEL_MAP.keys()),
        help=f"Select models to infer from: {list(MODEL_MAP.keys())}"
    )
    return parser.parse_args()


def extract_json_from_response(raw_text: str) -> dict:
    """Robust JSON extractor supporting markdown blocks, single quotes, trailing commas, and unclosed brackets."""
    default_res = {"skill": [], "knowledge": []}
    if not raw_text or not raw_text.strip():
        return default_res

    # 1. Strip reasoning chain tags if present
    cleaned_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    if "<think>" in cleaned_text:
        cleaned_text = cleaned_text.split("<think>")[-1]

    # 2. Extract JSON candidate string
    json_candidate = None
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned_text, re.DOTALL)
    if code_block:
        json_candidate = code_block.group(1)
    else:
        brace_match = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
        if brace_match:
            json_candidate = brace_match.group(0)

    # 3. Truncation recovery
    if not json_candidate and "{" in cleaned_text:
        truncated = cleaned_text[cleaned_text.find("{") :]
        last_comma = truncated.rfind(",")
        if last_comma != -1:
            json_candidate = truncated[:last_comma] + "]}"

    if not json_candidate:
        return default_res

    # 4. Multi-level parsing fallback
    parsed = None
    try:
        parsed = json.loads(json_candidate)
    except Exception:
        try:
            cleaned = re.sub(r",\s*([\]}])", r"\1", json_candidate)
            parsed = json.loads(cleaned)
        except Exception:
            try:
                parsed = ast.literal_eval(json_candidate)
            except Exception:
                pass

    if not isinstance(parsed, dict):
        return default_res

    skills_raw = (
        parsed.get("skill")
        or parsed.get("skills")
        or parsed.get("Skill")
        or parsed.get("Skills")
        or []
    )
    knowledge_raw = (
        parsed.get("knowledge")
        or parsed.get("knowledges")
        or parsed.get("Knowledge")
        or []
    )

    if not isinstance(skills_raw, list):
        skills_raw = [str(skills_raw)] if skills_raw else []
    if not isinstance(knowledge_raw, list):
        knowledge_raw = [str(knowledge_raw)] if knowledge_raw else []

    return {
        "skill": [str(s).strip() for s in skills_raw if str(s).strip()],
        "knowledge": [str(k).strip() for k in knowledge_raw if str(k).strip()],
    }


def infer_single_model(model_key: str, parsed_items: list):
    base_model_id = MODEL_MAP.get(model_key.lower(), model_key)
    clean_name = base_model_id.rstrip("/").split("/")[-1].lower().replace("-", "_")
    checkpoint_path = os.path.join(CHECKPOINTS_BASE_DIR, f"{clean_name}_fullshot")

    output_filename = f"raw_extracted_{clean_name}_fullshot.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    print(f"\n==================================================")
    print(f"[Inference Target] Key: {model_key} | Model: {base_model_id}")
    print(f"[LoRA Checkpoint]  {checkpoint_path}")
    print(f"[Output File]      {output_path}")
    print(f"==================================================")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Load Base Model and Mount LoRA
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE_MAP,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    # 3. Format prompts
    all_prompts = []
    for item in parsed_items:
        tokens = item.get("tokens", [])
        sentence = " ".join(tokens)
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"Text: {sentence}\nReturn JSON strictly."},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        all_prompts.append(prompt)

    total_samples = len(parsed_items)
    raw_records = []
    debug_count = 0

    # 4. Batched inference
    with torch.no_grad():
        for i in tqdm(range(0, total_samples, BATCH_SIZE), desc=f"[{clean_name} | full-shot]"):
            batch_prompts = all_prompts[i : i + BATCH_SIZE]
            batch_items = parsed_items[i : i + BATCH_SIZE]

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            input_lens = inputs.input_ids.shape[1]
            generated_ids = outputs[:, input_lens:]
            batch_responses = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            for item, resp in zip(batch_items, batch_responses):
                extracted = extract_json_from_response(resp)

                if debug_count < 2:
                    print(f"\n--- [{clean_name} Sample {debug_count + 1}] ---")
                    print(f"Input : {' '.join(item.get('tokens', []))}")
                    print(f"Parsed: {extracted}")
                    debug_count += 1

                raw_records.append(
                    {
                        "idx": item.get("idx"),
                        "tokens": item.get("tokens", []),
                        "source": item.get("source", "tech"),
                        "extracted": extracted,
                    }
                )

    # 5. Save outputs
    with open(output_path, "w", encoding="utf-8") as f_out:
        for record in raw_records:
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[Saved] Result file -> {output_path}")

    # 6. Memory Cleanup before next model
    del model
    del base_model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(TEST_EMPTY_PATH, "r", encoding="utf-8") as f:
        test_lines = [line.strip() for line in f if line.strip()]
    parsed_items = [json.loads(line) for line in test_lines]

    print(f"[Init] Total test samples loaded: {len(parsed_items)}")
    print(f"[Init] Selected models to infer: {args.models}")

    for model_key in args.models:
        infer_single_model(model_key, parsed_items)

    print("\n[Finished] All inferences across selected models completed.")


if __name__ == "__main__":
    main()
