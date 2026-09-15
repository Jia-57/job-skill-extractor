"""
Step 1: Convert BIO annotations directly into pure entity spans without prompts.
"""
import json
from pathlib import Path

# ================= Global Path Configurations =================
INPUT_TRAIN_PATH = "/home/tu/tu_tu/tu_zxois72/26ss/NER_Job/Data/raw/train_merged.json"
OUTPUT_NER_PATH = "/home/tu/tu_tu/tu_zxois72/26ss/NER_Job/Data/raw/train_entities.jsonl"
# ==============================================================


def bio_to_entities(tokens: list, tags: list) -> list:
    """Align tokens with BIO tags and extract continuous entity strings."""
    entities = []
    current_entity = []
S
    for token, tag in zip(tokens, tags):
        if tag == "B":
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []
            current_entity.append(token)
        elif tag == "I":
            if current_entity:
                current_entity.append(token)
            else:
                # Fault tolerance: treat an orphaned 'I' tag as the start of an entity
                current_entity.append(token)
        else:  # 'O' tag
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []

    if current_entity:
        entities.append(" ".join(current_entity))

    return entities


def main():
    Path(OUTPUT_NER_PATH).parent.mkdir(parents=True, exist_ok=True)
    converted_count = 0

    with open(INPUT_TRAIN_PATH, "r", encoding="utf-8") as f_in, open(
        OUTPUT_NER_PATH, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            tokens = item.get("tokens", [])
            sentence = " ".join(tokens)

            tags_skill = item.get("tags_skill", [])
            tags_knowledge = item.get("tags_knowledge", [])

            skills = bio_to_entities(tokens, tags_skill)
            knowledges = bio_to_entities(tokens, tags_knowledge)

            # Clean record without hardcoded prompts
            clean_record = {
                "idx": item.get("idx"),
                "source": item.get("source", "tech"),
                "tokens": tokens,
                "sentence": sentence,
                "entities": {
                    "skill": skills,
                    "knowledge": knowledges,
                },
            }

            f_out.write(json.dumps(clean_record, ensure_ascii=False) + "\n")
            converted_count += 1

    print(f"[Done] Converted {converted_count} samples to pure NER format.")
    print(f"[Saved] Output File -> {OUTPUT_NER_PATH}")


if __name__ == "__main__":
    main()