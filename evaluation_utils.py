import os
import time
import typing

from openai import RateLimitError

from tqdm.auto import tqdm
from rag_helper import RAGBase


def calc_price(usage, input_price_per_million=0.96, output_price_per_million=3.84):
    input_cost = (usage.input_tokens / 1_000_000) * input_price_per_million
    output_cost = (usage.output_tokens / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def calc_total_price(usages, input_price_per_million=0.96, output_price_per_million=3.84):
    total_cost = 0.0

    for usage in usages:
        cost = calc_price(usage, input_price_per_million=input_price_per_million, output_price_per_million=output_price_per_million)
        total_cost = total_cost + cost["total_cost"]

    return total_cost


def llm_structured(client, instructions, user_prompt, output_type, model="gpt-5.4-mini"):
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_prompt}
    ]

    response = client.responses.create(
        model=model,
        input=messages,
        max_output_tokens=int(os.getenv("JUDGE_MAX_TOKENS", "512")),
    )

    raw_text = response.output_text.strip()
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    raw_text = raw_text.strip()

    try:
        parsed = output_type.model_validate_json(raw_text)
    except Exception:
        # model ignored the JSON format: look for one of the allowed verdict words
        # (longest first, so "PARTLY_CORRECT" isn't read as "CORRECT")
        allowed = typing.get_args(output_type.model_fields["relevance"].annotation)
        for verdict in sorted(allowed, key=len, reverse=True):
            if verdict in raw_text.upper():
                parsed = output_type(relevance=verdict, explanation=raw_text)
                break
        else:
            raise

    return parsed, response.usage


def llm_structured_retry(
    client,
    instructions,
    user_prompt,
    output_type,
    model="gpt-5.4-mini",
    max_retries=3,
):
    for attempt in range(max_retries):
        try:
            return llm_structured(
                client,
                instructions,
                user_prompt,
                output_type,
                model=model,
            )
        except RateLimitError:
            raise  # the OpenAI client already retried; waiting longer here only blocks the app
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


class RAGWithUsage(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        self.usages = []
        self.last_usage = None

    def llm(self, prompt):
        input_messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages,
            max_output_tokens=int(os.getenv("ANSWER_MAX_TOKENS", "1024")),
        )

        self.last_usage = response.usage
        self.usages.append(response.usage)

        return response.output_text

    def total_cost(self):
        return calc_total_price(self.usages)


def map_progress(pool, seq, f):
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            result = future.result()
            results.append(result)

    return results
