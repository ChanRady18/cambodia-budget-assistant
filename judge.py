"""
Two checks on every answer:

1. check_numbers()      -- code, no LLM. Every number in the answer must appear in the
                           budget documents the answer was built from.
2. evaluate_relevance() -- LLM judge. Reads the question, the same documents, and the
                           answer, and grades it CORRECT / PARTLY_CORRECT / INCORRECT.
"""
import os
import re
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from evaluation_utils import llm_structured_retry

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

VERDICTS = ("CORRECT", "PARTLY_CORRECT", "INCORRECT")


class JudgeVerdict(BaseModel):
    # field is still called "relevance" so the feedback table and dashboards don't change
    relevance: Literal["CORRECT", "PARTLY_CORRECT", "INCORRECT"]
    explanation: str


judge_instructions = """
You check answers from a Cambodian state budget assistant.
You get the QUESTION, the SOURCE DOCUMENTS the assistant was given, and its ANSWER.

Grade the answer:
- "CORRECT": answers the question, and every number, year, ministry and
  planned/actual label matches the source documents. Also CORRECT if the documents
  do not contain the answer and the assistant says it doesn't know.
- "PARTLY_CORRECT": right numbers but incomplete, or mixes up planned vs actual,
  or adds claims not in the documents.
- "INCORRECT": any number that contradicts the documents, wrong ministry or year,
  or an invented answer when the documents don't contain it.

The answer may be in Khmer or English; Khmer digits (០-៩) equal 0-9.

Respond ONLY with raw JSON (no markdown):
{"relevance": "CORRECT", "explanation": "one or two sentences"}
""".strip()

judge_prompt = """
QUESTION: {question}

SOURCE DOCUMENTS:
{context}

ANSWER: {answer}
""".strip()


# ------------------------------------------------------------------ number check

KH_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text):
    """All numbers in a text, normalised: Khmer digits -> 0-9, commas and spaces removed."""
    text = text.translate(KH_DIGITS)
    text = re.sub(r"(?<=\d)[   ](?=\d{3}\b)", "", text)   # 1 520 000 -> 1520000
    found = set()
    for m in NUMBER_RE.findall(text):
        n = m.replace(",", "").rstrip(".")
        if n:
            found.add(n)
    return found


def check_numbers(answer, context, min_digits=3):
    """
    Return (status, unsupported_numbers).
    status: "OK" (all numbers found in the documents), "MISMATCH", or "NO_NUMBERS".
    Small numbers (< min_digits digits, e.g. "5" in "top 5") are ignored.
    """
    in_answer = {n for n in _numbers(answer) if len(n.replace(".", "")) >= min_digits}
    if not in_answer:
        return "NO_NUMBERS", []
    in_context = _numbers(context)
    missing = sorted(n for n in in_answer if n not in in_context)
    return ("MISMATCH" if missing else "OK"), missing


# ------------------------------------------------------------------ LLM judge

def evaluate_relevance(question, answer, context="", client=None, model=None, api_key=None, base_url=None):
    """context = the prompt / documents the answer was generated from (record.prompt)."""
    if client is None:
        load_dotenv()
        client = OpenAI(
            base_url=base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )
    model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)

    # the RAG prompt starts with "QUESTION: ...", keep only the documents part
    if "CONTEXT:" in context:
        context = context.split("CONTEXT:", 1)[1].strip()

    prompt = judge_prompt.format(question=question, context=context or "(none)", answer=answer)
    result, usage = llm_structured_retry(client, judge_instructions, prompt, JudgeVerdict, model=model)
    return result.relevance, result.explanation


if __name__ == "__main__":
    context = "MPWT planned expenditure for 2024: 1,520,000,000,000 riel. Source: budget-law-2024.pdf"
    print(check_numbers("The 2024 budget is 1,520,000,000,000 riel.", context))   # OK
    print(check_numbers("The 2024 budget is 1,500,000,000,000 riel.", context))   # MISMATCH
    print(evaluate_relevance("What is MPWT's 2024 budget?", "1,500,000,000,000 riel.", context))
