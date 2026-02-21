"""
process_dataset.py — Clean, format, and split the generated dataset
====================================================================

Takes the raw generated JSONL and produces train/val splits ready
for fine-tuning.

What this does:
  1. QUALITY FILTER — remove too-short, too-long, or malformed entries
  2. FORMAT — convert to chat message format (what the model actually sees)
  3. SPLIT — 90/10 train/val split with stratification by category
  4. STATS — token count estimates, length distributions

Why chat format matters:
  Modern LLMs are trained on chat-formatted data, not raw text.
  Each model family has its own chat template:
    - Qwen: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>...
    - Llama: <|begin_of_text|><|start_header_id|>system<|end_header_id|>...
    - Mistral: [INST] ... [/INST]

  We store data in a generic messages format:
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]

  The tokenizer's apply_chat_template() handles the conversion
  to model-specific tokens during training. This means our dataset
  works with ANY model — just swap the tokenizer.

Usage:
  python process_dataset.py --input dataset_raw.jsonl --output data/
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter


# ==================== SYSTEM PROMPT ====================

SYSTEM_PROMPT = (
    "You are a knowledgeable jewelry industry assistant. "
    "You provide accurate, detailed answers about gemstones, "
    "jewelry settings, metals, pricing, care, and e-commerce. "
    "Your responses are practical and helpful for both consumers "
    "and jewelry professionals."
)


# ==================== QUALITY FILTERS ====================

def quality_filter(item: dict) -> tuple[bool, str]:
    """
    Check if a Q&A pair meets quality standards.

    Returns (pass, reason) — reason is empty if passed.

    We're strict here. Bad training data = bad model.
    Better to have 400 great examples than 600 mediocre ones.
    """
    instruction = item.get("instruction", "")
    output = item.get("output", "")

    # Length checks
    if len(instruction) < 15:
        return False, "instruction too short"
    if len(output) < 80:
        return False, "output too short (< 80 chars)"
    if len(output) > 3000:
        return False, "output too long (> 3000 chars)"
    if len(instruction) > 500:
        return False, "instruction too long"

    # Content checks
    output_lower = output.lower()
    if "as an ai" in output_lower or "i cannot" in output_lower:
        return False, "contains AI refusal language"
    if "i'm sorry" in output_lower and "i don't" in output_lower:
        return False, "contains hedging/refusal"
    if output.count("*") > 10:
        return False, "excessive markdown formatting"

    # Structural checks
    if instruction.strip() == output.strip():
        return False, "instruction equals output"

    return True, ""


# ==================== FORMAT CONVERSION ====================

def to_messages(item: dict) -> dict:
    """
    Convert Alpaca format to chat messages format.

    Input:
      {"instruction": "...", "input": "...", "output": "..."}

    Output:
      {"messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ]}

    If "input" is non-empty, it's appended to the instruction
    (simulates providing additional context with the question).
    """
    user_content = item["instruction"]
    if item.get("input", "").strip():
        user_content += "\n\n" + item["input"]

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": item["output"]},
        ],
        "category": item.get("category", "unknown"),
    }


# ==================== TRAIN/VAL SPLIT ====================

def stratified_split(
    data: list[dict],
    val_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Split data into train/val with category stratification.

    Why stratify:
      If you random split, you might put all "pricing" examples
      in training and none in validation. Then your val loss doesn't
      tell you how well the model learns pricing.

      Stratification ensures each category is proportionally
      represented in both splits.
    """
    rng = random.Random(seed)

    # Group by category
    by_cat = {}
    for item in data:
        cat = item.get("category", "unknown")
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(item)

    train, val = [], []

    for cat, items in by_cat.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))  # at least 1 per category
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)

    return train, val


# ==================== TOKEN ESTIMATION ====================

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4


def dataset_stats(data: list[dict], label: str = ""):
    """Print dataset statistics."""
    if not data:
        print(f"  {label}: empty")
        return

    total_tokens = 0
    lengths = []
    categories = Counter()

    for item in data:
        msgs = item.get("messages", [])
        full_text = " ".join(m["content"] for m in msgs)
        tokens = estimate_tokens(full_text)
        total_tokens += tokens
        lengths.append(tokens)
        categories[item.get("category", "unknown")] += 1

    avg_tokens = total_tokens // len(data)
    max_tokens = max(lengths)
    min_tokens = min(lengths)

    print(f"\n  {label}:")
    print(f"    Examples: {len(data)}")
    print(f"    Total tokens: ~{total_tokens:,}")
    print(f"    Avg tokens/example: ~{avg_tokens}")
    print(f"    Range: {min_tokens} - {max_tokens} tokens")
    print(f"    Categories: {dict(categories.most_common())}")


# ==================== MAIN ====================

def process_dataset(
    input_path: str = "dataset_raw.jsonl",
    output_dir: str = "data",
    val_ratio: float = 0.10,
) -> tuple[list[dict], list[dict]]:
    """
    Full processing pipeline: load → filter → format → split → save.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load raw data
    print(f"Loading from {input_path}...")
    raw_data = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw_data.append(json.loads(line))
    print(f"  Loaded {len(raw_data)} raw examples")

    # Quality filter
    print(f"\nFiltering...")
    filtered = []
    reject_reasons = Counter()
    for item in raw_data:
        passed, reason = quality_filter(item)
        if passed:
            filtered.append(item)
        else:
            reject_reasons[reason] += 1

    print(f"  Passed: {len(filtered)} / {len(raw_data)}")
    if reject_reasons:
        print(f"  Rejected:")
        for reason, count in reject_reasons.most_common():
            print(f"    {reason}: {count}")

    # Convert to chat format
    print(f"\nConverting to chat format...")
    formatted = [to_messages(item) for item in filtered]

    # Split
    print(f"\nSplitting (val_ratio={val_ratio})...")
    train, val = stratified_split(formatted, val_ratio=val_ratio)
    print(f"  Train: {len(train)} | Val: {len(val)}")

    # Stats
    dataset_stats(train, "Train set")
    dataset_stats(val, "Validation set")

    # Save
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    for path, data in [(train_path, train), (val_path, val)]:
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"\n  Saved {len(data)} examples to {path}")

    # Also save a combined version for inspection
    all_path = output_dir / "all.jsonl"
    with open(all_path, "w") as f:
        for item in formatted:
            f.write(json.dumps(item) + "\n")

    return train, val


def main():
    parser = argparse.ArgumentParser(description="Process jewelry Q&A dataset")
    parser.add_argument("--input", default="dataset_raw.jsonl")
    parser.add_argument("--output", default="data")
    parser.add_argument("--val-ratio", type=float, default=0.10)
    args = parser.parse_args()

    print(f"{'='*50}")
    print(f"  Dataset Processor")
    print(f"{'='*50}")

    process_dataset(
        input_path=args.input,
        output_dir=args.output,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
