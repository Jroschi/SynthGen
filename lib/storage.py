import json
import os
from typing import Dict, Any

from .utils import setup_logger

logger = setup_logger("storage")

def append_jsonl(path: str, record: Dict[str, Any]):
    """
    Safely appends a JSON object to a newline.
    """
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Failed to append to {path}: {e}")

def convert_to_arrow(infile: str, outfile: str):
    """
    Converts a JSONL dataset to Apache Arrow format using huggingface datasets.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Huggingface 'datasets' library is required to convert to Arrow.")
        return

    if not os.path.exists(infile):
        logger.error(f"Input file {infile} does not exist.")
        return

    try:
        logger.info(f"Loading dataset from {infile}")
        dataset = load_dataset("json", data_files=infile, split="train")
        logger.info(f"Saving to disk at {outfile}")
        dataset.save_to_disk(outfile)
        logger.info("Conversion complete.")
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
