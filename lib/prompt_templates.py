import os
from typing import List

DEFAULT_STYLE = "Always respond in the style of a high medieval English aristocrat."

def compose_prompt(user_prompt: str, style_instruction: str = DEFAULT_STYLE) -> str:
    """
    Composes the full prompt to send to the teacher model.
    """
    return f"You are to respond *always* in the style: \"{style_instruction}\"\n\nUser prompt:\n{user_prompt}\n\nRespond with a single answer, do not include meta commentary."

def load_seed_prompts(path: str) -> List[str]:
    """
    Loads base prompts from a text file, ignoring empty lines.
    """
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts
