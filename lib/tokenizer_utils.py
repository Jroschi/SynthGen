import os
from transformers import AutoTokenizer

# Fallback basic tokenizer if needed, but prefer HF ones
_tokenizer = None

def get_tokenizer(model_id="mistralai/Mistral-7B-Instruct-v0.2"):
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            # Provide a fallback if network is down or model not found
            from transformers import GPT2TokenizerFast
            try:
                _tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            except Exception:
                # ultimate fallback if we have no access to hf hub
                class DumbTokenizer:
                    def encode(self, text):
                        # Approximate: 1 token = 4 chars roughly
                        return list(range(max(1, len(text) // 4)))
                _tokenizer = DumbTokenizer()
    return _tokenizer

def count_tokens(text: str, tokenizer=None) -> int:
    """
    Counts the number of tokens in the given text using the provided tokenizer.
    Uses get_tokenizer() by default.
    """
    if not text:
        return 0
    if tokenizer is None:
        tokenizer = get_tokenizer()

    encoded = tokenizer.encode(text)
    return len(encoded)
