import sys
from pathlib import Path

# Dynamically add the project root directory (NER_Job) to the Python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Training.prompts import build_chat_messages
import argparse
import ast
import gc
import json
import os
import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Built-in supported model dictionary mapping
MODEL_MAP = {
    "qwen": "Qwen/Qwen3-8B",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}


def extract_json_from_response(raw_text: str) -> dict:
    """Robust JSON extractor supporting markdown blocks, single quotes,
    trailing commas, unclosed brackets, and plural/singular keys."""
    default_res = {"skill": [], "knowledge": []}
    if not raw_text or not raw_text.strip():
        return default_res

    # 1. Filter out the thinking chain content if present
    cleaned_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    if "<think>" in cleaned_text:
        cleaned_text = cleaned_text.split("<think>")[-1]

    # 2. Match JSON code blocks or brace regions
    json_candidate = None
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned_text, re.DOTALL)
    if code_block:
        json_candidate = code_block.group(1)
    else:
        brace_match = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
        if brace_match:
            json_candidate = brace_match.group(0)

    # 3. Fallback logic: if truncated by max_new_tokens missing right brackets, try to auto-complete
    if not json_candidate and "{" in cleaned_text:
        truncated = cleaned_text[cleaned_text.find("{") :]
        last_comma = truncated.rfind(",")
        if last_comma != -1:
            json_candidate = truncated[:last_comma] + "]}"

    if not json_candidate:
        return default_res

    # 4. Multi-level deserialization parsing (Standard JSON -> Remove trailing commas -> Python literal ast)
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

    # 5. Compatible with singular/plural keys and capitalized initials
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


def run_inference(
    model,
    tokenizer,
    model_id: str,
    shot_type: str,
    test_lines: list,
    output_dir: str,
    max_new_tokens: int,
    batch_size: int = 32,
):
    clean_name = model_id.rstrip("/").split("/")[-1].lower().replace("-", "_")
    clean_shot = shot_type.replace("-", "")
    output_filename = f"raw_extracted_{clean_name}_{clean_shot}.json"
    output_path = os.path.join(output_dir, output_filename)

    print(f"\nRunning: Model={model_id} | Mode={shot_type} | BatchSize={batch_size}")
    print(f"Target : {output_path}")

    # 1. Core setting: Decoder-only generation models must perform left padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Parse data and construct prompts
    parsed_items = [json.loads(line) for line in test_lines]
    all_prompts = []
    for item in parsed_items:
        tokens = item.get("tokens", [])
        sentence = " ".join(tokens)
        messages = build_chat_messages(sentence, shot_type=shot_type)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        all_prompts.append(prompt)

    total_samples = len(parsed_items)
    raw_records = []
    sample_print_count = 0  # Print only the first 3 items for validation to prevent massive log I/O blocking

    # 3. Feed into GPU in mini-batches
    with torch.no_grad():
        for i in tqdm(range(0, total_samples, batch_size), desc=f"[{clean_name} | {shot_type}]"):
            batch_prompts = all_prompts[i : i + batch_size]
            batch_items = parsed_items[i : i + batch_size]

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
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

                if sample_print_count < 3:
                    print(f"\n--- [Debug Sample {sample_print_count + 1}] ---")
                    print(f"Input Text   : {' '.join(item.get('tokens', []))}")
                    print(f"Raw Response : {repr(resp)}")
                    print(f"Parsed Result: {extracted}")
                    print("---------------------------------")
                    sample_print_count += 1

                raw_records.append(
                    {
                        "idx": item.get("idx"),
                        "tokens": item.get("tokens", []),
                        "source": item.get("source", "unknown"),
                        "extracted": extracted,
                    }
                )

    with open(output_path, "w", encoding="utf-8") as f_out:
        for record in raw_records:
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1: Extract entities using LLM"
    )
    parser.add_argument(
        "--model",
        "--models",
        dest="model",
        type=str,
        default="qwen",
        choices=["qwen", "llama", "Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct"],
        help="Select the model to run: qwen or llama",
    )
    parser.add_argument(
        "--shot_types",
        nargs="+",
        default=["zero-shot", "few-shot"],
        choices=["zero-shot", "few-shot"],
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for parallel inference on A100",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=192,
        help="Max output tokens for entity JSON extraction",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="/home/tu/tu_tu/tu_zxois72/26ss/NER_Job/Data/raw/test_empty.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/tu/tu_tu/tu_zxois72/26ss/NER_Job/Data/llm_outputs",
    )
    parser.add_argument("--device_map", type=str, default="auto")

    args = parser.parse_args()

    model_id = MODEL_MAP.get(args.model.lower(), args.model)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.test_path, "r", encoding="utf-8") as f:
        test_lines = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(test_lines)} instances from {args.test_path}")
    print(f"Selected Model: {model_id} (key: {args.model})")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=args.device_map,
    )
    model.eval()

    # Eliminate sampling hyperparameter warnings during greedy decoding
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    for shot_type in args.shot_types:
        run_inference(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            shot_type=shot_type,
            test_lines=test_lines,
            output_dir=args.output_dir,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()