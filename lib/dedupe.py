import hashlib
from typing import Set

try:
    from fuzzywuzzy import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

class Deduplicator:
    def __init__(self, exact_only: bool = False, fuzzy_threshold: int = 90):
        self.seen_hashes: Set[str] = set()
        self.seen_texts: list[str] = []
        self.exact_only = exact_only
        self.fuzzy_threshold = fuzzy_threshold

    def get_hash(self, text: str) -> str:
        """
        Returns the SHA256 hash of a normalized string.
        """
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        """
        Checks if the exact or fuzzy-matched text has been seen before.
        """
        if not text.strip():
            return True

        h = self.get_hash(text)
        if h in self.seen_hashes:
            return True

        if not self.exact_only and FUZZY_AVAILABLE:
            normalized = text.strip().lower()
            for seen_text in self.seen_texts:
                if fuzz.token_set_ratio(normalized, seen_text) > self.fuzzy_threshold:
                    return True

        self.add(text, h)
        return False

    def add(self, text: str, text_hash: str = None):
        """
        Registers text as seen.
        """
        if text_hash is None:
            text_hash = self.get_hash(text)

        self.seen_hashes.add(text_hash)

        if not self.exact_only and FUZZY_AVAILABLE:
            self.seen_texts.append(text.strip().lower())
