import argparse
import glob
import json
import os


def align_spans_to_bio(tokens: list, spans: list) -> list:
    """Align entity spans back to tokens to generate B, I, O sequence."""
    tags = ["O"] * len(tokens)
    sorted_spans = sorted(spans, key=lambda x: len(x), reverse=True)

    for span in sorted_spans:
        span_clean = span.strip().lower()
        if not span_clean:
            continue
        span_tokens = span_clean.split()
        span_len = len(span_tokens)

        for i in range(len(tokens) - span_len + 1):
            window = [t.lower() for t in tokens[i : i + span_len]]
            if window == span_tokens:
                if tags[i] == "O":
                    tags[i] = "B"
                    for j in range(i + 1, i + span_len):
                        tags[j] = "I"
                break
    return tags


def convert_files(data_dir: str, output_dir: str, pattern: str):
    os.makedirs(output_dir, exist_ok=True)
    raw_files = glob.glob(os.path.join(data_dir, pattern))
    
    if not raw_files:
        print(f"No files matching '{pattern}' found in {data_dir}")
        return

    for file_path in raw_files:
        out_filename = os.path.basename(file_path).replace(
            "raw_extracted_", "pred_"
        )
        out_filepath = os.path.join(output_dir, out_filename)

        print(f"Converting {os.path.basename(file_path)} -> {out_filename}")

        converted_records = []
        with open(file_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                if not line.strip():
                    continue
                item = json.loads(line)

                tokens = item.get("tokens", [])
                extracted = item.get("extracted", {})

                record = {
                    "idx": item.get("idx"),
                    "tokens": tokens,
                    "tags_skill": align_spans_to_bio(
                        tokens, extracted.get("skill", [])
                    ),
                    "tags_knowledge": align_spans_to_bio(
                        tokens, extracted.get("knowledge", [])
                    ),
                    "source": item.get("source", "unknown"),
                }
                converted_records.append(record)

        with open(out_filepath, "w", encoding="utf-8") as f_out:
            for rec in converted_records:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\nConversion complete.")


if __name__ == "__main__":
   

    DATA_DIR = "/home/tu/tu_tu/tu_zxois72/26ss/NER_Job/Data/llm_outputs"
    OUTPUT_DIR = "/home/tu/tu_tu/tu_zxois72/26ss/NER_Job/Data/llm_outputs/converted"
    FILE_PATTERN = "raw_extracted_*.json"  # 也可以写特定的文件名，比如 "raw_extracted_qwen3_8b_zeroshot.json"

    convert_files(data_dir=DATA_DIR, output_dir=OUTPUT_DIR, pattern=FILE_PATTERN)
