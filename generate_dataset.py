"""
generate_dataset.py — Synthetic Jewelry Q&A Dataset Generator
==============================================================

WHY SYNTHETIC DATA:
  You rarely have a pre-existing dataset for domain-specific fine-tuning.
  The standard approach in 2024-2025:
    1. Write 2-3 high-quality examples per category (seeds.json)
    2. Use a strong LLM to generate hundreds more in the same style
    3. Clean, deduplicate, and validate
    4. Human review a sample for quality

  This is how Alpaca, Vicuna, and most instruction-tuned models were built.
  The key insight: a strong model (Claude, GPT-4) can generate training
  data that makes a small model (Qwen 3B) much better at a specific domain.

  IS THIS "CHEATING"?
    No — this is knowledge distillation. The big model has broad knowledge
    but is expensive to run. The small model learns a narrow domain deeply
    and runs cheaply. In production, you serve the small model. This is a
    legitimate and widely-used technique (see: Orca, Phi, Alpaca papers).

DATASET FORMAT:
  We use the Alpaca instruction format:
    {
      "instruction": "the question or task",
      "input": "" (optional additional context),
      "output": "the detailed answer"
    }

  This maps to chat format during training:
    <|user|> {instruction}\n{input}
    <|assistant|> {output}

Usage:
  python generate_dataset.py --provider claude --count 500
  python generate_dataset.py --provider openai --count 500
  python generate_dataset.py --dry-run   # preview prompts without API calls
"""

import json
import os
import random
import time
import argparse
from pathlib import Path


# ==================== GENERATION PROMPT ====================

GENERATION_PROMPT = """You are a jewelry industry expert creating training data for an AI assistant that specializes in jewelry, gemstones, and e-commerce.

CATEGORY: {category_name}
DESCRIPTION: {category_description}

Here are example Q&A pairs for this category:

{examples}

Generate {count} NEW Q&A pairs in the same category. Each pair should:
1. Cover a DIFFERENT topic than the examples (no repeats)
2. Be specific and detailed — include numbers, grades, technical terms
3. Be practically useful for someone shopping for or selling jewelry
4. Match the depth and style of the examples above
5. Vary the difficulty — some basic, some expert-level

Return ONLY valid JSON — an array of objects with "instruction" and "output" keys.
Do NOT include any text before or after the JSON array.

Example format:
[
  {{"instruction": "question here", "output": "detailed answer here"}},
  {{"instruction": "another question", "output": "another detailed answer"}}
]"""


def format_examples(examples: list[dict]) -> str:
    """Format seed examples for the prompt."""
    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(f"Example {i}:")
        parts.append(f"Q: {ex['instruction']}")
        parts.append(f"A: {ex['output']}")
        parts.append("")
    return "\n".join(parts)


# ==================== LLM CALLERS ====================

def call_claude(prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Call Anthropic Claude API."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0.8,  # higher for diversity
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def call_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI-compatible API."""
    import openai

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        temperature=0.8,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ==================== PARSING & CLEANING ====================

def parse_response(text: str) -> list[dict]:
    """
    Extract JSON array from LLM response.

    LLMs sometimes wrap JSON in markdown code blocks or add preamble.
    We handle that gracefully.
    """
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Find the JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        print(f"  ⚠ No JSON array found in response")
        return []

    json_str = text[start:end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  ⚠ JSON parse error: {e}")
        # Try to fix common issues
        json_str = json_str.replace("'", '"')
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return []

    # Validate structure
    valid = []
    for item in data:
        if isinstance(item, dict) and "instruction" in item and "output" in item:
            # Basic quality checks
            if len(item["output"]) < 50:
                continue  # too short
            if len(item["instruction"]) < 10:
                continue  # too short
            # Ensure input field exists
            item.setdefault("input", "")
            valid.append(item)

    return valid


def deduplicate(dataset: list[dict]) -> list[dict]:
    """
    Remove near-duplicate questions.

    Uses normalized question text for matching. Not perfect,
    but catches obvious duplicates from multiple generation rounds.
    """
    seen = set()
    unique = []

    for item in dataset:
        # Normalize: lowercase, strip punctuation, collapse whitespace
        key = item["instruction"].lower().strip()
        key = "".join(c for c in key if c.isalnum() or c == " ")
        key = " ".join(key.split())

        if key not in seen:
            seen.add(key)
            unique.append(item)

    removed = len(dataset) - len(unique)
    if removed > 0:
        print(f"  Removed {removed} duplicates")

    return unique


# ==================== MAIN GENERATOR ====================

def generate_dataset(
    seeds_path: str = "seeds.json",
    output_path: str = "dataset_raw.jsonl",
    provider: str = "claude",
    per_category: int = 15,
    dry_run: bool = False,
) -> list[dict]:
    """
    Generate full dataset from seed examples.

    Args:
        seeds_path: path to seeds.json
        output_path: where to save generated data
        provider: "claude" or "openai"
        per_category: how many Q&A pairs to generate per category
        dry_run: if True, print prompts without calling API
    """
    with open(seeds_path) as f:
        seeds = json.load(f)

    categories = seeds["categories"]
    all_data = []

    # Include seed examples in the dataset
    for cat in categories:
        for ex in cat["examples"]:
            ex.setdefault("input", "")
            ex["category"] = cat["name"]
            all_data.append(ex)

    print(f"Seeds: {len(all_data)} examples across {len(categories)} categories")

    caller = call_claude if provider == "claude" else call_openai

    for cat in categories:
        print(f"\n{'─'*50}")
        print(f"Category: {cat['name']} (generating {per_category})")

        prompt = GENERATION_PROMPT.format(
            category_name=cat["name"],
            category_description=cat["description"],
            examples=format_examples(cat["examples"]),
            count=per_category,
        )

        if dry_run:
            print(f"  [DRY RUN] Prompt length: {len(prompt)} chars")
            continue

        try:
            print(f"  Calling {provider}...", end=" ", flush=True)
            response = caller(prompt)
            print("done")

            items = parse_response(response)
            for item in items:
                item["category"] = cat["name"]
            all_data.extend(items)
            print(f"  Generated {len(items)} pairs (total: {len(all_data)})")

        except Exception as e:
            print(f"  ⚠ Error: {e}")
            continue

        # Rate limiting
        time.sleep(1)

    if dry_run:
        print(f"\n[DRY RUN] Would generate ~{per_category * len(categories)} pairs")
        return []

    # Deduplicate
    print(f"\nDeduplicating...")
    all_data = deduplicate(all_data)

    # Save
    output = Path(output_path)
    with open(output, "w") as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")

    print(f"\nSaved {len(all_data)} examples to {output}")

    # Stats
    cats = {}
    for item in all_data:
        c = item.get("category", "unknown")
        cats[c] = cats.get(c, 0) + 1
    print(f"\nPer category:")
    for c, count in sorted(cats.items()):
        print(f"  {c}: {count}")

    return all_data


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description="Generate jewelry Q&A dataset")
    parser.add_argument("--seeds", default="seeds.json", help="Path to seeds.json")
    parser.add_argument("--output", default="dataset_raw.jsonl", help="Output file")
    parser.add_argument("--provider", default="claude", choices=["claude", "openai"])
    parser.add_argument("--per-category", type=int, default=15, help="Q&A pairs per category")
    parser.add_argument("--dry-run", action="store_true", help="Preview prompts only")
    args = parser.parse_args()

    print(f"{'='*50}")
    print(f"  Jewelry Q&A Dataset Generator")
    print(f"  Provider: {args.provider}")
    print(f"  Per category: {args.per_category}")
    print(f"{'='*50}")

    generate_dataset(
        seeds_path=args.seeds,
        output_path=args.output,
        provider=args.provider,
        per_category=args.per_category,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
