# Fine-tuning a Jewelry Domain LLM with QLoRA

Fine-tune Qwen 2.5 3B into a jewelry domain expert using QLoRA on Google Colab (free T4 GPU).

## What You'll Learn

| Concept | Where |
|---------|-------|
| Synthetic data generation | `generate_dataset.py` |
| Instruction tuning format | `process_dataset.py` |
| 4-bit quantization (BitsAndBytes) | `train.py` |
| LoRA adapters (PEFT) | `train.py` |
| Chat templates & tokenization | `train.py` |
| Training loop & hyperparameters | `train.py` |
| Base vs fine-tuned evaluation | `eval_finetune.py` |
| LLM-as-judge evaluation | `eval_finetune.py` |

## Quick Start

### Step 1: Generate Dataset (local or Colab)

```bash
# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY

# Generate ~100 Q&A pairs (15 per category × 7 categories + seeds)
python generate_dataset.py --provider claude --per-category 15

# Process into train/val splits
python process_dataset.py --input dataset_raw.jsonl --output data/
```

### Step 2: Train on Colab

Upload these files to Colab: `seeds.json`, `data/train.jsonl`, `data/val.jsonl`, `train.py`

```python
# Cell 1: Install
!pip install -q torch transformers datasets peft bitsandbytes accelerate trl

# Cell 2: Upload data
from google.colab import files
# Upload train.jsonl and val.jsonl to data/ directory

# Cell 3: Run training
!python train.py --data-dir data --output-dir output --epochs 3
```

Training takes ~30-60 min on T4.

### Step 3: Evaluate

```bash
python eval_finetune.py --adapter output/final_adapter

# With LLM judge (optional, needs API key)
python eval_finetune.py --adapter output/final_adapter --judge claude
```

## Project Structure

```
finetune-jewelry/
├── seeds.json              # Hand-written seed Q&A pairs (7 categories)
├── generate_dataset.py     # LLM-powered synthetic data generation
├── process_dataset.py      # Clean, format, train/val split
├── train.py                # QLoRA fine-tuning (Colab-ready)
├── eval_finetune.py        # Base vs fine-tuned comparison
├── requirements.txt        # Dependencies
├── data/                   # Generated after processing
│   ├── train.jsonl
│   └── val.jsonl
└── output/                 # Generated after training
    └── final_adapter/      # LoRA adapter weights (~50MB)
```

## Key Decisions

**Why Qwen 2.5 3B?** Fits on T4 (16GB) with QLoRA. Strong instruction-following baseline. Larger context window than Llama 3B alternatives.

**Why QLoRA over full fine-tuning?** Full fine-tune of a 3B model needs ~24GB VRAM. QLoRA needs ~8GB. Same quality (proven in the QLoRA paper).

**Why synthetic data?** No jewelry Q&A dataset exists. Synthetic generation from a strong model (Claude/GPT-4) is the standard approach — this is how Alpaca, Orca, and Phi were built.

**Why 500 examples?** For domain adaptation (not teaching new capabilities), 200-1000 high-quality examples is sufficient. Quality >>> quantity. The QLoRA paper shows strong results with just 1000 examples.

## Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| LoRA rank | 32 | Good balance of capacity vs efficiency |
| LoRA alpha | 64 | 2× rank is standard scaling |
| Learning rate | 2e-4 | Standard for LoRA fine-tuning |
| Epochs | 3 | Enough to learn domain, low overfit risk |
| Batch size | 4 × 4 = 16 effective | Fits T4 VRAM with gradient accumulation |
| Max length | 1024 tokens | Covers our Q&A pairs with headroom |
| Quantization | NF4 + double quant | Optimal for normally-distributed weights |
