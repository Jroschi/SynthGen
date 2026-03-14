import requests
import time
import random
import os
import json
from typing import List, Dict, Any
from .utils import setup_logger

logger = setup_logger("ollama_client")

def generate_batch(
    prompts: List[str],
    config: Dict[str, Any],
    max_retries: int = 5,
    base_backoff_sec: float = 2.0
) -> List[str]:
    """
    Given a list of prompts, calls the Ollama endpoint.
    Assuming the default /api/generate endpoint for Ollama which takes one prompt at a time.
    For batching, we will call them sequentially or use threading if needed.
    """
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    api_key = os.environ.get("OLLAMA_API_KEY", "")

    endpoint = f"{url.rstrip('/')}/api/generate"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    responses = []

    for idx, prompt in enumerate(prompts):
        payload = {
            "model": config.get("model", "mistral-3-14b"),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.get("temperature", 0.7),
                "top_p": config.get("top_p", 0.95),
                "num_predict": config.get("max_new_tokens", 512),
            }
        }

        attempt = 0
        success = False
        while attempt < max_retries and not success:
            try:
                resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                responses.append(data.get("response", "").strip())
                success = True
            except requests.exceptions.RequestException as e:
                attempt += 1
                if attempt >= max_retries:
                    logger.error(f"Failed prompt after {max_retries} retries: {e}")
                    responses.append("") # Append empty string on final failure
                else:
                    sleep_time = base_backoff_sec * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Request failed: {e}. Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)

    return responses
