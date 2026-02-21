"""
train.py — QLoRA Fine-tuning on Google Colab
==============================================

This script is structured for Google Colab. Each section maps to a
notebook cell. Copy sections between # ═══ CELL markers into Colab cells.

WHAT IS QLoRA:
  Full fine-tuning updates ALL model weights — billions of parameters.
  That needs hundreds of GB of VRAM. Way beyond a T4.

  QLoRA (Quantized Low-Rank Adaptation) makes this feasible:
    1. QUANTIZE the base model to 4-bit (shrinks 7B model from 14GB to ~4GB)
    2. Freeze all original weights (they don't change)
    3. Add tiny LoRA adapter matrices to each attention layer
    4. Train ONLY the adapters (~1-2% of total parameters)

  Result: you get 95%+ of full fine-tuning quality at a fraction of the
  memory cost. A 3B model trains comfortably on a T4 (16GB VRAM).

  Key papers:
    - LoRA: Hu et al., 2021 (https://arxiv.org/abs/2106.09685)
    - QLoRA: Dettmers et al., 2023 (https://arxiv.org/abs/2305.14314)

Hardware: Google Colab T4 (16GB VRAM) — free tier works
Model: Qwen 2.5 3B Instruct
Training time: ~30-60 min on T4

Usage in Colab:
  1. Upload seeds.json, generate_dataset.py, process_dataset.py, train.py
  2. Run dataset generation (or upload pre-generated data/)
  3. Run training cells in order
  4. Download the adapter from output/

Usage locally:
  python train.py --data-dir data --output-dir output --epochs 3
"""

# ═══════════════════════════════════════════════════════════════
# CELL 1: Install dependencies
# ═══════════════════════════════════════════════════════════════

# !pip install -q torch transformers datasets peft bitsandbytes accelerate trl

# ═══════════════════════════════════════════════════════════════
# CELL 2: Check GPU
# ═══════════════════════════════════════════════════════════════

def check_gpu():
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {gpu}")
        print(f"VRAM: {vram:.1f} GB")

        if vram < 14:
            print("⚠ Less than 14GB VRAM — use Qwen 2.5 1.5B instead of 3B")
        return True
    else:
        print("⚠ No GPU detected! Go to Runtime → Change runtime type → T4 GPU")
        return False


# ═══════════════════════════════════════════════════════════════
# CELL 3: Load model with 4-bit quantization
# ═══════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    """
    Load the base model in 4-bit precision with BitsAndBytes.

    WHY 4-BIT:
      A 3B parameter model in fp16 needs ~6GB VRAM just for weights.
      In 4-bit, it needs ~1.5GB. This leaves plenty of room for
      optimizer states, gradients, and activations during training.

      BitsAndBytes uses NF4 (NormalFloat 4-bit) quantization with
      double quantization. NF4 is information-theoretically optimal
      for normally-distributed weights (which neural network weights
      approximately are).

    WHY NOT 8-BIT:
      4-bit works just as well for LoRA fine-tuning (shown in QLoRA paper)
      and uses half the memory. The LoRA adapters are still in fp16,
      so the actual trained parameters have full precision.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",            # NormalFloat4 — optimal for weights
        bnb_4bit_compute_dtype=torch.float16,  # Compute in fp16 for speed
        bnb_4bit_use_double_quant=True,        # Quantize the quantization constants too
    )

    print(f"Loading {model_name} in 4-bit...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",          # Automatically place layers on GPU
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Memory check
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded: {mem:.1f} GB VRAM used")
    print(f"Vocab size: {len(tokenizer)}")

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════
# CELL 4: Attach LoRA adapters
# ═══════════════════════════════════════════════════════════════

def attach_lora(model, rank: int = 32, alpha: int = 64, dropout: float = 0.05):
    """
    Add LoRA adapter matrices to the model.

    HOW LoRA WORKS:
      For a weight matrix W (d × d), instead of updating all d² parameters,
      LoRA adds two small matrices: A (d × r) and B (r × d) where r << d.

      Forward pass: output = W·x + (B·A)·x

      W is frozen. Only A and B are trained.
      Total trainable params: 2 × d × r instead of d².
      For d=2048, r=32: 131K vs 4.2M params per layer — 97% reduction.

    KEY HYPERPARAMETERS:
      rank (r): Dimensionality of the adapter. Higher = more capacity
        but more parameters. 16-64 is typical. 32 is a good default.
        The QLoRA paper found r=64 works well but r=16 is usually enough.

      alpha: Scaling factor. The adapter output is scaled by alpha/rank.
        Higher alpha = adapters have more influence on the output.
        Rule of thumb: alpha = 2 × rank (so alpha=64 for rank=32).

      dropout: Regularization on adapter inputs. 0.05-0.1 is typical.
        Prevents overfitting on small datasets.

      target_modules: Which layers get adapters. For transformer models,
        applying to all attention projections (q, k, v, o) and the
        dense projections in MLP is standard.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    # Prepare model for k-bit training
    # This handles gradient checkpointing and layer norm dtype
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",                   # Don't train bias terms
        task_type=TaskType.CAUSAL_LM,
        target_modules=[               # Which layers get LoRA adapters
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",       # MLP
        ],
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameter count
    model.print_trainable_parameters()
    # Expected: ~1-2% of total parameters are trainable

    return model


# ═══════════════════════════════════════════════════════════════
# CELL 5: Load and tokenize dataset
# ═══════════════════════════════════════════════════════════════

def load_dataset(data_dir: str = "data", tokenizer=None, max_length: int = 1024):
    """
    Load train/val JSONL and tokenize using the model's chat template.

    WHY apply_chat_template:
      Each model family expects a specific format:
        Qwen: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...
        Llama: <|start_header_id|>user<|end_header_id|>...

      apply_chat_template() handles this automatically based on the
      tokenizer config. This means our data prep works for ANY model.

    WHY max_length=1024:
      Our Q&A pairs are 100-500 tokens typically. 1024 gives headroom
      without wasting memory on padding. Longer sequences = more VRAM
      per batch = smaller batch size = slower training.
    """
    from datasets import Dataset

    def load_jsonl(path):
        items = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items

    data_dir = Path(data_dir)
    train_data = load_jsonl(data_dir / "train.jsonl")
    val_data = load_jsonl(data_dir / "val.jsonl")

    print(f"Loaded {len(train_data)} train, {len(val_data)} val examples")

    def tokenize(examples):
        texts = []
        for msgs in examples["messages"]:
            # apply_chat_template converts messages to model-specific format
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        # Tokenize the formatted text
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        # For causal LM, labels = input_ids (model predicts next token)
        # We mask padding tokens in labels with -100 (ignored in loss)
        tokenized["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in ids]
            for ids in tokenized["input_ids"]
        ]

        return tokenized

    # Create HF datasets
    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    # Remove category column before tokenizing
    if "category" in train_ds.column_names:
        train_ds = train_ds.remove_columns(["category"])
        val_ds = val_ds.remove_columns(["category"])

    print("Tokenizing...")
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["messages"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["messages"])

    print(f"Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")

    # Show a sample
    sample = tokenizer.decode(train_ds[0]["input_ids"], skip_special_tokens=False)
    print(f"\nSample (first 300 chars):\n{sample[:300]}...")

    return train_ds, val_ds


# ═══════════════════════════════════════════════════════════════
# CELL 6: Training
# ═══════════════════════════════════════════════════════════════

def train(
    model,
    tokenizer,
    train_ds,
    val_ds,
    output_dir: str = "output",
    epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation: int = 4,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.05,
    logging_steps: int = 10,
):
    """
    Train with QLoRA using HuggingFace Trainer.

    KEY HYPERPARAMETERS:
      epochs: 2-5 for fine-tuning. 3 is safe default.
        Too many = overfitting (model memorizes training data).
        Too few = underfitting (model hasn't learned the domain).
        Watch val loss — if it starts going UP while train loss goes down,
        you're overfitting.

      batch_size: Limited by VRAM. 4 works on T4 with 3B model.
        Effective batch = batch_size × gradient_accumulation = 16.
        Larger effective batch = more stable gradients.

      learning_rate: 1e-4 to 3e-4 is typical for LoRA fine-tuning.
        Higher than full fine-tuning because we're only updating small adapters.
        2e-4 is the standard starting point.

      warmup_ratio: Fraction of training to linearly increase LR.
        0.03-0.10 prevents early instability. 0.05 = 5% warmup.

      gradient_accumulation: Simulates larger batch without more VRAM.
        4 steps × batch_size 4 = effective batch of 16.

    WHAT TO WATCH:
      - train/loss should decrease steadily
      - eval/loss should decrease then plateau (or slightly increase)
      - If eval/loss increases early → reduce epochs or increase data
      - If train/loss doesn't decrease → increase learning rate
    """
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",         # Cosine decay — smooth LR reduction
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,                  # L2 regularization
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=50,                      # Evaluate every 50 steps
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,                 # Keep only 2 best checkpoints
        load_best_model_at_end=True,        # Load best checkpoint when done
        metric_for_best_model="eval_loss",
        greater_is_better=False,            # Lower eval_loss = better
        fp16=True,                          # Mixed precision training
        report_to="none",                   # Disable wandb/tensorboard
        dataloader_pin_memory=True,
        gradient_checkpointing=True,        # Trade compute for memory
        optim="paged_adamw_8bit",           # 8-bit AdamW — saves ~2GB VRAM
    )

    # Data collator handles dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    print(f"\nTraining config:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size} × {gradient_accumulation} = {batch_size * gradient_accumulation} effective")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Total steps: ~{len(train_ds) * epochs // (batch_size * gradient_accumulation)}")
    print(f"\nStarting training...")

    result = trainer.train()

    print(f"\nTraining complete!")
    print(f"  Train loss: {result.training_loss:.4f}")
    print(f"  Runtime: {result.metrics['train_runtime']:.0f}s")

    return trainer


# ═══════════════════════════════════════════════════════════════
# CELL 7: Save adapter
# ═══════════════════════════════════════════════════════════════

def save_adapter(model, tokenizer, output_dir: str = "output/final_adapter"):
    """
    Save ONLY the LoRA adapter weights.

    WHY NOT SAVE THE FULL MODEL:
      The base model is ~6GB. The LoRA adapter is ~50MB.
      When you deploy:
        1. Load the base model (same for everyone)
        2. Load YOUR adapter on top (tiny, fast)

      This is how production LoRA works — you share the base model
      across many fine-tunes and swap adapters for different domains.
      One base Qwen 3B + different adapters for jewelry, medical, legal, etc.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Check adapter size
    adapter_files = list(output_dir.glob("adapter_model*"))
    total_size = sum(f.stat().st_size for f in adapter_files)
    print(f"Adapter saved to {output_dir}")
    print(f"Adapter size: {total_size / 1e6:.1f} MB")
    print(f"(Compare to base model: ~6000 MB)")


# ═══════════════════════════════════════════════════════════════
# CELL 8: Inference — test the fine-tuned model
# ═══════════════════════════════════════════════════════════════

def test_model(model, tokenizer, questions: list[str] = None):
    """
    Test the fine-tuned model with sample questions.

    Compares base model knowledge vs fine-tuned responses.
    """
    import torch

    if questions is None:
        questions = [
            "What does VS2 clarity mean for a diamond?",
            "What are the pros and cons of a tension setting?",
            "How much should I expect to pay for a 1 carat diamond engagement ring?",
            "What is the difference between natural and lab-grown emeralds?",
            "Write a product description for 14K yellow gold hoop earrings with diamonds.",
            "A customer says their white gold ring is turning yellow. What happened?",
        ]

    system = (
        "You are a knowledgeable jewelry industry assistant. "
        "You provide accurate, detailed answers about gemstones, "
        "jewelry settings, metals, pricing, care, and e-commerce."
    )

    print(f"\n{'='*60}")
    print(f"  Testing fine-tuned model")
    print(f"{'='*60}")

    for q in questions:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": q},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"\n{'─'*60}")
        print(f"Q: {q}")
        print(f"\nA: {answer}")


# ═══════════════════════════════════════════════════════════════
# CELL 9: Full pipeline (or CLI entry point)
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="QLoRA fine-tuning")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()

    # GPU check
    has_gpu = check_gpu()
    if not has_gpu:
        print("Aborting — need GPU for training")
        return

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)

    if args.test_only:
        # Load existing adapter and test
        from peft import PeftModel
        adapter_path = Path(args.output_dir) / "final_adapter"
        if adapter_path.exists():
            print(f"Loading adapter from {adapter_path}...")
            model = PeftModel.from_pretrained(model, str(adapter_path))
        test_model(model, tokenizer)
        return

    # Attach LoRA
    model = attach_lora(model, rank=args.lora_rank)

    # Load data
    train_ds, val_ds = load_dataset(args.data_dir, tokenizer, args.max_length)

    # Train
    trainer = train(
        model, tokenizer, train_ds, val_ds,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Save adapter
    save_adapter(model, tokenizer, f"{args.output_dir}/final_adapter")

    # Test
    test_model(model, tokenizer)

    print(f"\n{'='*60}")
    print(f"  DONE!")
    print(f"  Adapter saved to: {args.output_dir}/final_adapter/")
    print(f"  To test later: python train.py --test-only")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
