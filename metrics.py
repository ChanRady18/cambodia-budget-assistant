import os
import time
from dataclasses import dataclass, field
from datetime import datetime

from rag_helper import RAGBase, answer_max_tokens


@dataclass
class LLMCallRecord:
    model: str
    prompt: str
    instructions: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)


# $ per 1M tokens (input, output). Groq prices from console.groq.com/docs/models (Oct 2026).
PRICES = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "qwen/qwen3.8-27b": (0.80, 4.00),
}


def calculate_cost(model, usage, input_price=None, output_price=None):
    if input_price is None and os.getenv("INPUT_TOKEN_PRICE"):
        input_price = float(os.getenv("INPUT_TOKEN_PRICE"))
        output_price = float(os.getenv("OUTPUT_TOKEN_PRICE", "0"))
    if input_price is None and model in PRICES:
        input_price, output_price = PRICES[model]
    if input_price is not None and output_price is not None:
        return (usage.input_tokens * input_price + usage.output_tokens * output_price) / 1_000_000
    cost = 0
    if "gpt-5.4-mini" in model:
        cost = (usage.input_tokens * 0.15 + usage.output_tokens * 0.60) / 1_000_000
    elif "qwen" in model:
        cost = (usage.input_tokens * 0.96 + usage.output_tokens * 3.84) / 1_000_000
    return cost


class RAGWithMetrics(RAGBase):

    def __init__(self, *args, input_price_per_million=None, output_price_per_million=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_price_per_million = input_price_per_million
        self.output_price_per_million = output_price_per_million
        self.last_call: LLMCallRecord = None

    def llm(self, prompt):
        start_time = time.time()
        response = self._call_llm(prompt)
        response_time = time.time() - start_time
        self._log_response(prompt, response, response_time)
        return response.output_text

    def _call_llm(self, prompt):
        input_messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]
        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages,
            max_output_tokens=answer_max_tokens(),
        )
        return response

    def _log_response(self, prompt, response, response_time):
        usage = response.usage
        cost = calculate_cost(self.model, usage, self.input_price_per_million, self.output_price_per_million)

        call_record = LLMCallRecord(
            model=self.model,
            prompt=prompt,
            instructions=self.instructions,
            answer=response.output_text,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            response_time=response_time,
            cost=cost,
        )
    
        print(call_record)
        self.last_call = call_record
