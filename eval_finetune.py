"""
eval_finetune.py — Compare base model vs fine-tuned on jewelry domain
======================================================================

This is the payoff — proof that fine-tuning actually worked.

Two types of evaluation:
  1. QUALITATIVE: Ask the same questions to base and fine-tuned model,
     see the difference side-by-side.
  2. QUANTITATIVE: Score answers on domain accuracy, specificity,
     and helpfulness using a held-out test set.

Why both matter:
  Qualitative shows the "vibe" — fine-tuned answers feel more expert.
  Quantitative gives you numbers for your resume/portfolio.

Usage:
  python eval_finetune.py --adapter output/final_adapter
  python eval_finetune.py --adapter output/final_adapter --judge claude
"""

import json
import time
import argparse
from pathlib import Path


# ==================== EVAL QUESTIONS ====================
# These are held-out — NOT in the training data.
# Mix of difficulty levels to test different aspects.

EVAL_QUESTIONS = [
    {
        "question": "What's the difference between a cushion cut and a radiant cut diamond?",
        "category": "gemstones",
        "difficulty": "medium",
        "key_points": ["facet pattern", "shape", "brilliance style", "rounded vs square corners"],
    },
    {
        "question": "Is platinum better than white gold for an engagement ring?",
        "category": "metals",
        "difficulty": "medium",
        "key_points": ["durability", "price", "hypoallergenic", "weight", "maintenance"],
    },
    {
        "question": "What does 'eye-clean' mean when shopping for diamonds?",
        "category": "terminology",
        "difficulty": "easy",
        "key_points": ["no visible inclusions", "naked eye", "SI1/VS2 range", "value advantage"],
    },
    {
        "question": "How can I tell if a ruby has been heat treated?",
        "category": "treatments",
        "difficulty": "hard",
        "key_points": ["common treatment", "silk dissolution", "certification", "price impact", "disclosure"],
    },
    {
        "question": "Write a product description for a men's tungsten wedding band with brushed finish.",
        "category": "product_descriptions",
        "difficulty": "medium",
        "key_points": ["scratch resistance", "weight", "cannot resize", "comfort fit", "hypoallergenic"],
    },
    {
        "question": "A customer wants to know why their sterling silver bracelet tarnished. How do you explain this?",
        "category": "customer_service",
        "difficulty": "easy",
        "key_points": ["sulfur reaction", "normal process", "cleaning method", "prevention tips", "storage"],
    },
    {
        "question": "What factors should I consider when buying a colored gemstone like a sapphire versus a diamond?",
        "category": "pricing",
        "difficulty": "hard",
        "key_points": ["different grading systems", "color primacy", "origin premium", "treatments standard", "no universal price list"],
    },
    {
        "question": "Explain the difference between pavé and micro-pavé settings.",
        "category": "settings",
        "difficulty": "medium",
        "key_points": ["stone size", "bead work", "durability", "sparkle density", "cost difference"],
    },
]


# ==================== GENERATE ANSWERS ====================

def generate_answer(model, tokenizer, question: str, max_tokens: int = 512) -> str:
    """Generate answer from the model."""
    import torch

    system = (
        "You are a knowledgeable jewelry industry assistant. "
        "You provide accurate, detailed answers about gemstones, "
        "jewelry settings, metals, pricing, care, and e-commerce."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ==================== SIMPLE SCORING ====================

def score_answer(answer: str, key_points: list[str]) -> dict:
    """
    Simple heuristic scoring — checks if answer covers key points.

    This is NOT a production eval — it's a rough proxy.
    For real scoring, use an LLM judge (see judge_with_llm below).
    """
    answer_lower = answer.lower()
    hits = []
    misses = []

    for point in key_points:
        # Check if any word from the key point appears
        point_words = point.lower().split()
        found = any(word in answer_lower for word in point_words if len(word) > 3)
        if found:
            hits.append(point)
        else:
            misses.append(point)

    return {
        "coverage": len(hits) / len(key_points) if key_points else 0,
        "hits": hits,
        "misses": misses,
        "length": len(answer),
        "word_count": len(answer.split()),
    }


# ==================== LLM JUDGE (optional) ====================

def judge_with_llm(question: str, base_answer: str, finetuned_answer: str, provider: str = "claude") -> dict:
    """
    Use a strong LLM to judge which answer is better.

    This is the standard approach in LLM evaluation (see: MT-Bench,
    AlpacaEval, Chatbot Arena). A strong model judges the quality
    of weaker models' outputs.

    Returns: {"winner": "base"|"finetuned"|"tie", "reasoning": "..."}
    """
    import os

    judge_prompt = f"""You are evaluating two AI responses about jewelry. Rate them on:
1. Domain accuracy (correct jewelry/gemstone facts)
2. Specificity (concrete details vs vague generalities)
3. Helpfulness (practical value for the reader)
4. Completeness (covers important aspects)

Question: {question}

--- Response A (Base Model) ---
{base_answer}

--- Response B (Fine-tuned Model) ---
{finetuned_answer}

Which response is better for a jewelry domain assistant? Reply with ONLY valid JSON:
{{"winner": "A" or "B" or "tie", "score_a": 1-10, "score_b": 1-10, "reasoning": "brief explanation"}}"""

    try:
        if provider == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                temperature=0,
                messages=[{"role": "user", "content": judge_prompt}],
            )
            text = response.content[0].text
        else:
            import openai
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=300,
                temperature=0,
                messages=[{"role": "user", "content": judge_prompt}],
            )
            text = response.choices[0].message.content

        # Parse JSON from response
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            result = json.loads(text[start:end + 1])
            # Map A/B back to base/finetuned
            if result.get("winner") == "A":
                result["winner"] = "base"
            elif result.get("winner") == "B":
                result["winner"] = "finetuned"
            return result

    except Exception as e:
        print(f"  Judge error: {e}")

    return {"winner": "error", "reasoning": "judge failed"}


# ==================== EVALUATION RUNNER ====================

def run_eval(
    base_model,
    finetuned_model,
    tokenizer,
    use_judge: str = None,
):
    """
    Run full evaluation: base vs fine-tuned on held-out questions.
    """
    results = []

    print(f"\n{'='*70}")
    print(f"  EVALUATION: Base vs Fine-tuned")
    print(f"  Questions: {len(EVAL_QUESTIONS)}")
    if use_judge:
        print(f"  LLM Judge: {use_judge}")
    print(f"{'='*70}")

    for i, q in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n{'─'*70}")
        print(f"  [{i}/{len(EVAL_QUESTIONS)}] {q['question'][:60]}...")
        print(f"  category={q['category']} difficulty={q['difficulty']}")

        # Generate from both models
        print(f"  Generating base answer...", end=" ", flush=True)
        base_answer = generate_answer(base_model, tokenizer, q["question"])
        print(f"done ({len(base_answer.split())} words)")

        print(f"  Generating fine-tuned answer...", end=" ", flush=True)
        ft_answer = generate_answer(finetuned_model, tokenizer, q["question"])
        print(f"done ({len(ft_answer.split())} words)")

        # Score both
        base_score = score_answer(base_answer, q["key_points"])
        ft_score = score_answer(ft_answer, q["key_points"])

        result = {
            "question": q["question"],
            "category": q["category"],
            "difficulty": q["difficulty"],
            "base_answer": base_answer,
            "finetuned_answer": ft_answer,
            "base_coverage": base_score["coverage"],
            "ft_coverage": ft_score["coverage"],
            "base_words": base_score["word_count"],
            "ft_words": ft_score["word_count"],
        }

        # LLM judge (optional)
        if use_judge:
            print(f"  Judging...", end=" ", flush=True)
            judgment = judge_with_llm(q["question"], base_answer, ft_answer, use_judge)
            result["judge_winner"] = judgment.get("winner", "error")
            result["judge_score_base"] = judgment.get("score_a", 0)
            result["judge_score_ft"] = judgment.get("score_b", 0)
            result["judge_reasoning"] = judgment.get("reasoning", "")
            print(f"winner={result['judge_winner']}")
            time.sleep(0.5)  # rate limiting

        results.append(result)

        # Show preview
        print(f"\n  BASE ({base_score['coverage']:.0%} coverage):")
        print(f"  {base_answer[:150]}...")
        print(f"\n  FINE-TUNED ({ft_score['coverage']:.0%} coverage):")
        print(f"  {ft_answer[:150]}...")

    # Aggregate
    print_eval_summary(results, use_judge)
    return results


def print_eval_summary(results: list[dict], use_judge: str = None):
    """Print evaluation summary."""
    print(f"\n{'='*70}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*70}")

    # Coverage comparison
    base_avg = sum(r["base_coverage"] for r in results) / len(results)
    ft_avg = sum(r["ft_coverage"] for r in results) / len(results)
    print(f"\n  Key point coverage (heuristic):")
    print(f"    Base model:      {base_avg:.1%}")
    print(f"    Fine-tuned:      {ft_avg:.1%}")
    print(f"    Improvement:     {ft_avg - base_avg:+.1%}")

    # Word count (detail level)
    base_words = sum(r["base_words"] for r in results) / len(results)
    ft_words = sum(r["ft_words"] for r in results) / len(results)
    print(f"\n  Average response length:")
    print(f"    Base model:      {base_words:.0f} words")
    print(f"    Fine-tuned:      {ft_words:.0f} words")

    # Judge results (if available)
    if use_judge and any("judge_winner" in r for r in results):
        wins = {"base": 0, "finetuned": 0, "tie": 0, "error": 0}
        judge_base_avg = 0
        judge_ft_avg = 0
        judged = 0

        for r in results:
            winner = r.get("judge_winner", "error")
            wins[winner] = wins.get(winner, 0) + 1
            if "judge_score_base" in r:
                judge_base_avg += r["judge_score_base"]
                judge_ft_avg += r["judge_score_ft"]
                judged += 1

        print(f"\n  LLM Judge results ({use_judge}):")
        print(f"    Base wins:       {wins['base']}")
        print(f"    Fine-tuned wins: {wins['finetuned']}")
        print(f"    Ties:            {wins['tie']}")
        if judged:
            print(f"    Avg score base:  {judge_base_avg/judged:.1f}/10")
            print(f"    Avg score FT:    {judge_ft_avg/judged:.1f}/10")

    # Per-difficulty breakdown
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            base_c = sum(r["base_coverage"] for r in subset) / len(subset)
            ft_c = sum(r["ft_coverage"] for r in subset) / len(subset)
            print(f"\n  {diff.upper()} questions ({len(subset)}):")
            print(f"    Base: {base_c:.1%}  |  Fine-tuned: {ft_c:.1%}  |  Δ = {ft_c-base_c:+.1%}")


# ==================== CLI ====================

def main():
    import torch

    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default="output/final_adapter")
    parser.add_argument("--judge", default=None, choices=["claude", "openai"],
                        help="Use LLM judge for scoring")
    parser.add_argument("--save", default="eval_results.json")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    # Load base model (quantized)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load fine-tuned (base + adapter)
    print(f"Loading adapter from {args.adapter}...")
    finetuned_model = PeftModel.from_pretrained(base_model, args.adapter)

    # Run eval
    results = run_eval(
        base_model=base_model,
        finetuned_model=finetuned_model,
        tokenizer=tokenizer,
        use_judge=args.judge,
    )

    # Save results
    with open(args.save, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()
