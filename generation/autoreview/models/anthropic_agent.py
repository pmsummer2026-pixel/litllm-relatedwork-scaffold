"""
Minimal Anthropic (Claude) API wrapper matching the interface used by
plan_based_generation.py, mirroring autoreview/models/langchain_openai_agent.py's
OpenAIAgent so it can be swapped in whenever --model_name starts with "claude".

Expected interface:
  - get_response(prompt) -> dict with keys "response", "total_cost",
    "prompt", "n_tokens"
  - get_state_dict() -> {"budget_spent": <float>}

Requires the ANTHROPIC_API_KEY environment variable to be set (e.g. as a
Codespaces/local secret). The key itself is never hardcoded or logged here.
"""

import os
import time

import anthropic

API_KEY = os.environ["ANTHROPIC_API_KEY"]

# Approximate USD price per million tokens, used only to estimate
# get_state_dict()'s "budget_spent". Update if Anthropic's pricing changes.
PRICE_PER_MILLION_TOKENS = {
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-20241022": {"input": 0.8, "output": 4.0},
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
}
DEFAULT_PRICE = {"input": 3.0, "output": 15.0}


class AnthropicAgent:
    """
    Main class for Anthropic (Claude) agents. Drop-in counterpart to
    OpenAIAgent for use in plan_based_generation.py.
    """

    def __init__(self, model_name="claude-3-5-sonnet-20241022", prompt_name="prompt_action_gpt", max_tokens=1024):
        super().__init__()
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=API_KEY)
        self.executed_actions = []
        self.budget_spent = 0
        print(f"Running the experiment for {model_name} using the Anthropic API")

    def get_state_dict(self):
        return {"budget_spent": self.budget_spent}

    def get_response(self, prompt, max_retries=8):
        system = "You are a helpful assistant."
        message = None
        attempt = 0
        while True:
            attempt += 1
            try:
                message = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except anthropic.RateLimitError as e:
                print(f"Rate limited: {e}")
                time.sleep(1.0)
            except anthropic.APIStatusError as e:
                print(f"Anthropic API error: {e}")
                if attempt >= max_retries:
                    raise
                time.sleep(0.5)
            except Exception as e:
                print(f"Exception occured as: {e}")
                if attempt >= max_retries:
                    raise
                time.sleep(0.5)

        response_text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )
        prices = PRICE_PER_MILLION_TOKENS.get(self.model_name, DEFAULT_PRICE)
        n_input = message.usage.input_tokens
        n_output = message.usage.output_tokens
        total_cost = (n_input / 1_000_000) * prices["input"] + (n_output / 1_000_000) * prices["output"]

        response_dict = {
            "response": response_text,
            "total_cost": total_cost,
            "prompt": system + "\n\n" + prompt,
            "n_tokens": n_input + n_output,
        }
        self.budget_spent += total_cost
        return response_dict

    def get_price(self):
        return PRICE_PER_MILLION_TOKENS.get(self.model_name, DEFAULT_PRICE)
