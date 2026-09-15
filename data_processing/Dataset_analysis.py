import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def extract_dataset_stats(file_path):
    posts = set()
    sentences = 0
    skill_count = 0
    know_count = 0
    skill_lengths = []
    know_lengths = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sentences += 1
            posts.add(item.get("idx"))

            tags_s = item.get("tags_skill", [])
            tags_k = item.get("tags_knowledge", [])

            # Parse Skill entity spans
            i = 0
            while i < len(tags_s):
                if tags_s[i] == "B":
                    skill_count += 1
                    length = 1
                    j = i + 1
                    while j < len(tags_s) and tags_s[j] == "I":
                        length += 1
                        j += 1
                    skill_lengths.append(length)
                    i = j
                else:
                    i += 1

            # Parse Knowledge entity spans
            i = 0
            while i < len(tags_k):
                if tags_k[i] == "B":
                    know_count += 1
                    length = 1
                    j = i + 1
                    while j < len(tags_k) and tags_k[j] == "I":
                        length += 1
                        j += 1
                    know_lengths.append(length)
                    i = j
                else:
                    i += 1

    return {
        "posts": len(posts),
        "sentences": sentences,
        "skills": skill_count,
        "knowledge": know_count,
        "skill_lens": skill_lengths,
        "know_lens": know_lengths
    }

def main():
    train_stats = extract_dataset_stats("Data/train_merged.json")
    test_stats = extract_dataset_stats("Data/test.json")

    # 1. Print statistical data directly to the terminal
    print("=" * 60)
    print(f"{'Dataset Split':<18} | {'Posts':<8} | {'Sentences':<10} | {'Skills':<8} | {'Knowledge':<10}")
    print("-" * 60)
    print(f"{'Train (+Dev)':<18} | {train_stats['posts']:<8} | {train_stats['sentences']:<10} | {train_stats['skills']:<8} | {train_stats['knowledge']:<10}")
    print(f"{'Test (Gold)':<18} | {test_stats['posts']:<8} | {test_stats['sentences']:<10} | {test_stats['skills']:<8} | {test_stats['knowledge']:<10}")
    print("=" * 60)

    # 2. Generate entity length distribution comparison plot (Violin Plot)
    plot_data = []
    for l in train_stats['skill_lens']:
        plot_data.append({"Split": "Train (+Dev)", "Entity Type": "Skill", "Length": l})
    for l in train_stats['know_lens']:
        plot_data.append({"Split": "Train (+Dev)", "Entity Type": "Knowledge", "Length": l})
    for l in test_stats['skill_lens']:
        plot_data.append({"Split": "Test (Gold)", "Entity Type": "Skill", "Length": l})
    for l in test_stats['know_lens']:
        plot_data.append({"Split": "Test (Gold)", "Entity Type": "Knowledge", "Length": l})

    df = pd.DataFrame(plot_data)

    plt.figure(figsize=(8, 5))
    sns.set_theme(style="whitegrid")
    sns.violinplot(
        data=df, 
        x="Split", 
        y="Length", 
        hue="Entity Type", 
        split=True, 
        inner="quart", 
        palette={"Skill": "#2b5c8f", "Knowledge": "#e07a5f"}
    )
    plt.title("Entity Span Length Distribution Comparison", fontsize=12, fontweight='bold')
    plt.xlabel("Dataset Split", fontsize=10)
    plt.ylabel("Span Length (Token Count)", fontsize=10)
    plt.legend(title="Entity Type")
    plt.tight_layout()
    
    output_path = "Results/entity_length_distribution.png"
    plt.savefig(output_path, dpi=300)
    print(f"[✓] Entity length distribution violin plot successfully saved to: {output_path}")

if __name__ == "__main__":
    main()