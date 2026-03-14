# SynthGen

Generate compact-but-sufficient prompt→response pairs by prompting your teacher model (via Ollama) with style instructions, collect, tokenize & filter, save as JSONL (or Arrow), and provide a CLI parameter to target dataset size for 3B / 8B / 14B student models to train a LoRA.

## Setup

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you have `requests`, `transformers`, `tokenizers`, `datasets`, `fuzzywuzzy`, `python-Levenshtein`, `tqdm` installed).*

2. Make sure you have [Ollama](https://ollama.com/) running.
   If using a remote host or different port, set the `OLLAMA_URL` environment variable or pass `--teacher-url`.

## Usage

### Generate Dataset

Generate a dataset targeted to 3B models (~2M tokens):
```bash
python generate_dataset.py generate \
  --target 3b \
  --style "Always answer as a high medieval English aristocrat, formal and archaic." \
  --teacher-url "http://localhost:11434" \
  --out-file data/style_3b.jsonl
```

Generate by explicit token budget:
```bash
python generate_dataset.py generate \
  --target-tokens 2000000 \
  --batch-size 8 \
  --out-file data/style_bytokens.jsonl
```

### Show Statistics

Show dataset stats:
```bash
python generate_dataset.py stats --in-file data/style_3b.jsonl
```

### Convert to Arrow

Convert the JSONL dataset to Apache Arrow format:
```bash
python generate_dataset.py convert \
  --in-file data/style_3b.jsonl \
  --out-file data/style_3b.arrow
```

### Sample a Validation Set

Create a small holdout evaluation set:
```bash
python generate_dataset.py sample \
  --in-file data/style_3b.jsonl \
  --n 200
```
