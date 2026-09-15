"""
Prompts and Few-Shot Demonstrations for SKILLSPAN Entity Extraction.
Aligned with the annotation guidelines from Zhang et al. (2022).
"""

import json

SYSTEM_INSTRUCTION = """You are an expert annotator for Named Entity Recognition (NER) on job postings based on the SKILLSPAN annotation schema.
Your task is to extract all "skill" and "knowledge" components from the given text.

### Definitions and Guidelines:
1. **Knowledge (KNOWLEDGE)**:
   - Facts, principles, programming languages, technologies, tools, certifications, degrees, languages, and business domains that someone possesses or learns (e.g., "Python", "Docker", "Bachelor Degree", "English", "supply chain").
   - Rule-of-thumb: Knowledge is something possessed that cannot physically execute on its own.
   - Certifications/Licenses include extra words: "CSCS card", "full driving license".
   - Vague trigger verbs (e.g., "work with AWS", "follow code of practice") should NOT be included in the knowledge span; extract only "AWS", "code of practice".

2. **Skill (SKILL)**:
   - The ability to apply knowledge and execute actions to complete tasks (usually starts with a verb, e.g., "manage large sections of guests", "train new staff").
   - Soft skills and attitudes are treated as skills (e.g., "proactive", "team player", "attention to detail", "work independently").
   - Modal verbs ("can", "will", "should") and empty indicator phrases are excluded.
   - Keep spans concise: exclude irrelevant company-specific information appended at the end.

3. **Overlapping & Nested Entities**:
   - Knowledge components can be nested inside Skill components. Both spans must be extracted if applicable.

4. **Output Format Requirement**:
   - You MUST reply with ONLY a single valid JSON object.
   - Absolutely NO thinking process, NO conversational filler, NO Markdown block fences, and NO extra characters outside the JSON object.
   - Strictly use the schema:
     {"skill": ["..."], "knowledge": ["..."]}
   - If none found for a category, provide an empty list: []
"""

FEW_SHOT_EXAMPLES = [
    {
        "sentence": "You will thrive working in a Dev/Sec Ops culture.",
        "output": {
            "skill": ["working in a Dev/Sec Ops culture"],
            "knowledge": ["Dev/Sec Ops"],
        },
    },
    {
        "sentence": "The ability to manage large sections of guests.",
        "output": {
            "skill": ["manage large sections of guests"],
            "knowledge": [],
        },
    },
    {
        "sentence": (
            "Prior in-house experience with media, publishing or internet"
            " companies."
        ),
        "output": {
            "skill": [],
            "knowledge": ["media", "publishing", "internet companies"],
        },
    },
    {
        "sentence": (
            "Be inquisitive, proactive and proficient in Python and English."
        ),
        "output": {
            "skill": ["inquisitive", "proactive"],
            "knowledge": ["Python", "English"],
        },
    },
    {
        "sentence": (
            "Must hold a full uk driving licence and have hands-on experience in"
            " AWS infrastructure."
        ),
        "output": {
            "skill": ["hands-on"],
            "knowledge": ["full uk driving licence", "AWS infrastructure"],
        },
    },
]


def build_chat_messages(sentence: str, shot_type: str = "zero-shot") -> list:
    """Constructs clean message payloads conforming to Hugging Face chat templates."""
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]

    if shot_type == "few-shot":
        for ex in FEW_SHOT_EXAMPLES:
            messages.append({
                "role": "user",
                "content": f"Text: {ex['sentence']}\nReturn JSON strictly.",
            })
            # Must use standard json.dumps to ensure valid syntax
            messages.append({
                "role": "assistant",
                "content": json.dumps(ex["output"], ensure_ascii=False),
            })

    messages.append({
        "role": "user",
        "content": f"Text: {sentence}\nReturn JSON strictly.",
    })
    return messages