import sys

from dotenv import load_dotenv
from openai import OpenAI

from ingest import load_budget_data, build_index
from metrics import RAGWithMetrics
from db_save import save_conversation


import os

DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-max-2026-06-08"


def create_assistant(
    api_key=None,
    model=None,
    base_url=None,
    input_price_per_million=None,
    output_price_per_million=None
):
    load_dotenv()

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
    base_url = base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)

    documents = load_budget_data()
    index = build_index(documents)

    return RAGWithMetrics(
        index=index,
        llm_client=OpenAI(
            base_url=base_url,
            api_key=api_key
        ),
        model=model,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million
    )

if __name__ == "__main__":
    assistant = create_assistant()

    query = "How much is the 2024 budget of the Ministry of Public Works and Transport?"
    if len(sys.argv) > 1:
        query = sys.argv[1]

    answer = assistant.rag(query)
    print(answer)

    from rag_helper import detect_year
    save_conversation(assistant.last_call, query, detect_year(query) or "unknown")
