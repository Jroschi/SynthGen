import unittest
from lib.prompt_templates import compose_prompt
from lib.tokenizer_utils import count_tokens

class TestUtils(unittest.TestCase):
    def test_compose_prompt(self):
        user_prompt = "What is gravity?"
        style = "Like a pirate."
        composed = compose_prompt(user_prompt, style)
        self.assertIn("Like a pirate.", composed)
        self.assertIn("What is gravity?", composed)
        self.assertIn("Respond with a single answer", composed)

    def test_count_tokens(self):
        # A simple string should have >0 tokens
        text = "Hello world, this is a test."
        count = count_tokens(text)
        self.assertGreater(count, 0)
        self.assertLess(count, 20)  # rough sanity check

if __name__ == '__main__':
    unittest.main()
