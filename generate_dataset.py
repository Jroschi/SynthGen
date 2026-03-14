import argparse
import sys
import os
import uuid
import json
import time
import random
from typing import List, Dict, Any

from tqdm import tqdm

from lib.utils import setup_logger
from lib.tokenizer_utils import count_tokens, get_tokenizer
from lib.ollama_client import generate_batch
from lib.prompt_templates import compose_prompt, load_seed_prompts, DEFAULT_STYLE
from lib.dedupe import Deduplicator
from lib.storage import append_jsonl, convert_to_arrow

logger = setup_logger("generate_dataset")

# Target token guidelines based on requirements
TARGETS = {
    "3b": 2_000_000,
    "8b": 6_000_000,
    "14b": 12_000_000
}

def generate_subcommand(args):
    """
    Main loop for calling Ollama and accumulating data to hit token targets.
    """
    if args.target:
        target_tokens = TARGETS.get(args.target.lower(), args.target_tokens)
    else:
        target_tokens = args.target_tokens

    if target_tokens is None and args.examples is None:
        logger.error("Must specify --target, --target-tokens, or --examples.")
        sys.exit(1)

    logger.info(f"Starting generation. Target tokens: {target_tokens}, Target examples: {args.examples}")

    tokenizer = get_tokenizer()
    dedup = Deduplicator(exact_only=False, fuzzy_threshold=95)

    seed_prompts = load_seed_prompts("prompts_seed.txt")
    if not seed_prompts:
        logger.warning("No seed prompts found. Falling back to simple default questions.")
        seed_prompts = [
            "What is the meaning of life?",
            "Explain how a rainbow works.",
            "Write a short poem about the sea.",
            "How do I cook an egg?",
            "Can you write a sorting algorithm in Python?"
        ]

    config = {
        "model": args.teacher_model,
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": args.max_new_tokens
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    os.makedirs("raw", exist_ok=True)

    # Progress tracking
    cumulative_tokens = 0
    generated_examples = 0
    errors = 0

    pbar_tokens = tqdm(total=target_tokens, desc="Tokens", unit="tok") if target_tokens else None
    pbar_examples = tqdm(total=args.examples, desc="Examples", unit="ex") if args.examples else None

    # Continue until one of the conditions is met
    while True:
        if target_tokens and cumulative_tokens >= target_tokens:
            break
        if args.examples and generated_examples >= args.examples:
            break

        # Prepare batch
        batch_prompts = random.choices(seed_prompts, k=args.batch_size)
        composed_batch = [compose_prompt(p, args.style) for p in batch_prompts]

        # Call Teacher model
        responses = generate_batch(composed_batch, config, max_retries=args.max_retries)

        # Save raw dump for debugging
        timestamp = int(time.time())
        raw_dump = f"raw/{timestamp}_batch.json"
        try:
            with open(raw_dump, "w", encoding="utf-8") as rf:
                json.dump({"prompts": composed_batch, "responses": responses}, rf)
        except Exception:
            pass

        # Process and Filter
        for orig_prompt, composed, resp in zip(batch_prompts, composed_batch, responses):
            if not resp:
                errors += 1
                continue

            # Filters
            if len(resp.strip()) < 20: # skip very short answers
                continue

            combined_text = orig_prompt + "\n" + resp
            if dedup.is_duplicate(combined_text):
                continue

            # Count tokens
            p_toks = count_tokens(orig_prompt, tokenizer)
            r_toks = count_tokens(resp, tokenizer)
            tot_toks = p_toks + r_toks

            # Build record
            record = {
                "id": str(uuid.uuid4()),
                "prompt": orig_prompt,
                "response": resp,
                "style_instruction": args.style,
                "tokens_prompt": p_toks,
                "tokens_response": r_toks,
                "tokens_total": tot_toks,
            }

            # Append JSONL
            append_jsonl(args.out_file, record)

            # Update stats
            cumulative_tokens += tot_toks
            generated_examples += 1
            dedup.add(combined_text)

            if pbar_tokens:
                pbar_tokens.update(tot_toks)
            if pbar_examples:
                pbar_examples.update(1)

            if target_tokens and cumulative_tokens >= target_tokens:
                break
            if args.examples and generated_examples >= args.examples:
                break

    if pbar_tokens: pbar_tokens.close()
    if pbar_examples: pbar_examples.close()

    logger.info(f"Generation complete. Generated {generated_examples} examples with {cumulative_tokens} tokens total.")
    if errors > 0:
        logger.warning(f"Encountered {errors} failed generations/empty responses.")


def stats_subcommand(args):
    """
    Shows basic stats for a generated JSONL dataset.
    """
    if not os.path.exists(args.in_file):
        logger.error(f"Input file {args.in_file} not found.")
        sys.exit(1)

    total_examples = 0
    total_tokens = 0
    max_tokens = 0

    with open(args.in_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                total_examples += 1
                tot = record.get("tokens_total", 0)
                total_tokens += tot
                if tot > max_tokens:
                    max_tokens = tot
            except json.JSONDecodeError:
                pass

    print(f"Dataset Stats for {args.in_file}:")
    print(f"Total Examples: {total_examples}")
    print(f"Total Tokens: {total_tokens}")
    if total_examples > 0:
        print(f"Avg Tokens/Example: {total_tokens / total_examples:.2f}")
    print(f"Max Tokens/Example: {max_tokens}")


def convert_subcommand(args):
    """
    Converts JSONL to Arrow format.
    """
    convert_to_arrow(args.in_file, args.out_file)

def sample_subcommand(args):
    """
    Produces a small holdout eval set and removes those items from the main training set.
    """
    if not os.path.exists(args.in_file):
        logger.error(f"Input file {args.in_file} not found.")
        sys.exit(1)

    lines = []
    with open(args.in_file, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    if not lines:
        logger.error("Dataset is empty.")
        sys.exit(1)

    sample_size = min(len(lines), args.n)
    logger.info(f"Sampling {sample_size} examples from {len(lines)} total.")

    random.shuffle(lines)
    val_lines = lines[:sample_size]
    train_lines = lines[sample_size:]

    # Save validation
    val_file = args.in_file.replace(".jsonl", "_val.jsonl")
    with open(val_file, "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line + ('' if line.endswith('\n') else '\n'))

    # Overwrite train file
    train_file = args.in_file.replace(".jsonl", "_train.jsonl")
    with open(train_file, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + ('' if line.endswith('\n') else '\n'))

    logger.info(f"Saved {len(val_lines)} items to {val_file}")
    logger.info(f"Saved {len(train_lines)} items to {train_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate compact style LoRA datasets from an Ollama teacher model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser: generate
    parser_gen = subparsers.add_parser("generate", help="Generate dataset from Ollama")
    parser_gen.add_argument("--target", type=str, choices=["3b", "8b", "14b"], help="Target student model size for token budgets.")
    parser_gen.add_argument("--target-tokens", type=int, help="Exact target total tokens to collect.")
    parser_gen.add_argument("--examples", type=int, help="Exact target number of examples to collect.")
    parser_gen.add_argument("--style", type=str, default=DEFAULT_STYLE, help="Style instruction to use.")
    parser_gen.add_argument("--teacher-url", type=str, default="http://localhost:11434", help="Ollama endpoint URL.")
    parser_gen.add_argument("--teacher-model", type=str, default="mistral-3-14b", help="Teacher model name in Ollama.")
    parser_gen.add_argument("--batch-size", type=int, default=8, help="Number of prompts per batch.")
    parser_gen.add_argument("--out-file", type=str, default="data/dataset.jsonl", help="Output JSONL file path.")
    parser_gen.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser_gen.add_argument("--paraphrase-multiplier", type=int, default=1, help="Paraphrase multiplier (not yet implemented).")
    parser_gen.add_argument("--max-retries", type=int, default=5, help="Max retries for API requests.")
    parser_gen.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens for the model to generate.")

    # Subparser: stats
    parser_stats = subparsers.add_parser("stats", help="Show dataset stats")
    parser_stats.add_argument("--in-file", type=str, required=True, help="Input JSONL file.")

    # Subparser: convert
    parser_convert = subparsers.add_parser("convert", help="Convert JSONL to Arrow")
    parser_convert.add_argument("--in-file", type=str, required=True, help="Input JSONL file.")
    parser_convert.add_argument("--out-file", type=str, required=True, help="Output Arrow directory.")

    # Subparser: sample
    parser_sample = subparsers.add_parser("sample", help="Split off a holdout eval set")
    parser_sample.add_argument("--in-file", type=str, required=True, help="Input JSONL file.")
    parser_sample.add_argument("--n", type=int, default=200, help="Number of examples to hold out.")

    args = parser.parse_args()

    # Apply URL from args to env vars for simple integration
    if hasattr(args, "teacher_url") and args.teacher_url:
        os.environ["OLLAMA_URL"] = args.teacher_url

    random.seed(getattr(args, "seed", 42))

    if args.command == "generate":
        generate_subcommand(args)
    elif args.command == "stats":
        stats_subcommand(args)
    elif args.command == "convert":
        convert_subcommand(args)
    elif args.command == "sample":
        sample_subcommand(args)


if __name__ == "__main__":
    main()
