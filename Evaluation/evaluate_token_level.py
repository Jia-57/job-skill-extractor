import json
import os
from collections import defaultdict
from typing import Dict, Optional


def calc_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Calculate Precision, Recall, and F1-Score based on TP, FP, and FN"""
    p = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    r = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": round(p, 2), "recall": round(r, 2), "f1": round(f1, 2)}


def evaluate_with_source(
    gold_path: str,
    pred_path: str,
    output_dir: str = "./Results",
    model_name: Optional[str] = None
) -> Dict:
    if not os.path.exists(gold_path):
        raise FileNotFoundError(f"Gold annotation file not found: {gold_path}")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction result file not found: {pred_path}")

    # If model name is not specified, parse it automatically from the prediction filename (e.g., extract 'bert_base_cased' from 'pred_bert_base_cased.json')
    if not model_name:
        base_name = os.path.splitext(os.path.basename(pred_path))[0]
        if base_name.startswith("pred_"):
            model_name = base_name[len("pred_"):]
        else:
            model_name = base_name

    # Global statistics
    overall_stats = {
        "skill": {"tp": 0, "fp": 0, "fn": 0},
        "knowledge": {"tp": 0, "fp": 0, "fn": 0}
    }

    # Statistics by domain (source, e.g., house, tech, etc.)
    source_stats = defaultdict(lambda: {
        "skill": {"tp": 0, "fp": 0, "fn": 0},
        "knowledge": {"tp": 0, "fp": 0, "fn": 0}
    })

    with open(gold_path, 'r', encoding='utf-8') as f_gold, open(pred_path, 'r', encoding='utf-8') as f_pred:
        for line_no, (g_line, p_line) in enumerate(zip(f_gold, f_pred), start=1):
            if not g_line.strip() or not p_line.strip():
                continue

            gold = json.loads(g_line)
            pred = json.loads(p_line)

            src = gold.get("source", "unknown")
            tokens_len = len(gold.get("tokens", []))

            for task in ["skill", "knowledge"]:
                tag_key = f"tags_{task}"
                g_tags = gold.get(tag_key, [])
                p_tags = pred.get(tag_key, [])

                if len(g_tags) != tokens_len or len(p_tags) != tokens_len:
                    raise ValueError(f"Line {line_no}: Tag length does not match tokens length!")

                for g_t, p_t in zip(g_tags, p_tags):
                    g_is_ent = g_t in ("B", "I")
                    p_is_ent = p_t in ("B", "I")

                    if not g_is_ent and not p_is_ent:
                        continue

                    if g_is_ent and p_is_ent:
                        overall_stats[task]["tp"] += 1
                        source_stats[src][task]["tp"] += 1
                    elif not g_is_ent and p_is_ent:
                        overall_stats[task]["fp"] += 1
                        source_stats[src][task]["fp"] += 1
                    elif g_is_ent and not p_is_ent:
                        overall_stats[task]["fn"] += 1
                        source_stats[src][task]["fn"] += 1

    # Calculate global metrics
    res_skill = calc_prf(overall_stats["skill"]["tp"], overall_stats["skill"]["fp"], overall_stats["skill"]["fn"])
    res_know = calc_prf(overall_stats["knowledge"]["tp"], overall_stats["knowledge"]["fp"], overall_stats["knowledge"]["fn"])

    total_tp = overall_stats["skill"]["tp"] + overall_stats["knowledge"]["tp"]
    total_fp = overall_stats["skill"]["fp"] + overall_stats["knowledge"]["fp"]
    total_fn = overall_stats["skill"]["fn"] + overall_stats["knowledge"]["fn"]
    res_micro = calc_prf(total_tp, total_fp, total_fn)

    # 1. Print global metrics to console
    print("\n" + "=" * 75)
    print(f"Model Name: {model_name}")
    print(f"Prediction File: {pred_path}")
    print("=" * 75)
    print(f"{'Overall Category':<18} | {'Precision (%)':<14} | {'Recall (%)':<12} | {'F1-Score (%)':<10}")
    print("-" * 75)
    print(f"{'Skill':<18} | {res_skill['precision']:<14} | {res_skill['recall']:<12} | {res_skill['f1']:<10}")
    print(f"{'Knowledge':<18} | {res_know['precision']:<14} | {res_know['recall']:<12} | {res_know['f1']:<10}")
    print(f"{'Overall Micro':<18} | {res_micro['precision']:<14} | {res_micro['recall']:<12} | {res_micro['f1']:<10}")
    print("=" * 75)

    # 2. Print metrics by domain to console and format as JSON structure
    print("\n" + "-" * 75)
    print(f"{'Source Domain':<18} | {'Task':<10} | {'Precision (%)':<14} | {'Recall (%)':<12} | {'F1-Score (%)':<10}")
    print("-" * 75)

    formatted_source_stats = {}
    for src_name, s_data in source_stats.items():
        sk_prf = calc_prf(s_data["skill"]["tp"], s_data["skill"]["fp"], s_data["skill"]["fn"])
        kn_prf = calc_prf(s_data["knowledge"]["tp"], s_data["knowledge"]["fp"], s_data["knowledge"]["fn"])

        s_tp = s_data["skill"]["tp"] + s_data["knowledge"]["tp"]
        s_fp = s_data["skill"]["fp"] + s_data["knowledge"]["fp"]
        s_fn = s_data["skill"]["fn"] + s_data["knowledge"]["fn"]
        s_micro = calc_prf(s_tp, s_fp, s_fn)

        # Fully retain precision, recall, f1 and raw counts
        formatted_source_stats[src_name] = {
            "skill": {**sk_prf, "raw_counts": s_data["skill"]},
            "knowledge": {**kn_prf, "raw_counts": s_data["knowledge"]},
            "combined": {**s_micro, "raw_counts": {"tp": s_tp, "fp": s_fp, "fn": s_fn}}
        }

        print(f"[{src_name}]")
        print(f"   {'':<16} | {'Skill':<10} | {sk_prf['precision']:<14} | {sk_prf['recall']:<12} | {sk_prf['f1']:<10}")
        print(f"   {'':<16} | {'Knowledge':<10} | {kn_prf['precision']:<14} | {kn_prf['recall']:<12} | {kn_prf['f1']:<10}")
        print(f"   {'':<16} | {'Combined':<10} | {s_micro['precision']:<14} | {s_micro['recall']:<12} | {s_micro['f1']:<10}")
    print("-" * 75 + "\n")

    # Organize the final output result dictionary
    eval_result = {
        "model_name": model_name,
        "gold_path": gold_path,
        "pred_path": pred_path,
        "overall": {
            "skill": {**res_skill, "raw_counts": overall_stats["skill"]},
            "knowledge": {**res_know, "raw_counts": overall_stats["knowledge"]},
            "micro": {**res_micro, "raw_counts": {"tp": total_tp, "fp": total_fp, "fn": total_fn}}
        },
        "by_source": formatted_source_stats
    }

    # 3. Write to JSON file
    os.makedirs(output_dir, exist_ok=True)
    out_filename = f"{model_name}_results.json"
    out_filepath = os.path.join(output_dir, out_filename)

    with open(out_filepath, "w", encoding="utf-8") as f_out:
        json.dump(eval_result, f_out, ensure_ascii=False, indent=2)

    print(f"Evaluation complete, JSON results saved to: {out_filepath}\n")
    return eval_result


if __name__ == "__main__":
    GOLD_TEST = "/home/tu/tu_tu/tu_zxois72/26ss/NER_Job/Data/raw/test.json"
    #MODEL_PRED = "/home/tu/tu_tu/tu_zxois72/26ss/NER_Job/Data/predictions/pred_jobbert_base_cased.json"
    #MODEL_PRED = "/home/tu/tu_tu/tu_zxois72/26ss/NER_Job/Data/llm_outputs/converted/pred_qwen3_8b_fewshot.json"
    MODEL_PRED = "/home/tu/tu_tu/tu_zxois72/26ss/NER_Job/Data/llm_outputs/converted/pred_qwen3_8b_fullshot.json"
    #MODEL_PRED = "/home/tu/tu_tu/tu_zxois72/26ss/NER_Job/Data/llm_outputs/converted/pred_llama_3.1_8b_instruct_fullshot.json"  #pred_llama_3.1_8b_instruct_fewshot.json
    RESULTS_DIR = "/home/tu/tu_tu/tu_zxois72/26ss/NER_Job/Data/evaluation"

    if os.path.exists(MODEL_PRED):
        evaluate_with_source(GOLD_TEST, MODEL_PRED, output_dir=RESULTS_DIR)
    else:
        print(f"Prediction result file not found: {MODEL_PRED}")


