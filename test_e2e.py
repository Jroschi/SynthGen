import json
import os
import argparse
import sys
from unittest.mock import patch
from generate_dataset import main

# A simple script to simulate a mock run
if __name__ == '__main__':
    def mock_generate_batch(prompts, config, max_retries=5, base_backoff_sec=2.0):
        # return dummy responses based on prompt list
        return ["Hark! " + p for p in prompts]

    with patch("generate_dataset.generate_batch", mock_generate_batch):
        # We simulate sys.argv
        sys.argv = ["generate_dataset.py", "generate", "--target-tokens", "200", "--batch-size", "2", "--out-file", "data/test.jsonl"]
        main()
