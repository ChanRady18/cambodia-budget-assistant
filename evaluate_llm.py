"""
Compare answer quality of prompt variants on the hand-written test questions.

  python evaluate_llm.py            # all 40 questions (2 LLM calls per question per variant)
  python evaluate_llm.py --limit 10 # quicker

For every question and prompt variant: retrieve -> answer -> number check -> LLM judge.
Writes evaluation/llm_results.csv and prints a Markdown table to paste into the README.
Needs OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL in .env (e.g. Groq + openai/gpt-oss-120b).
"""
import argparse
import csv
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from ingest import BASE_DIR, build_index, load_budget_data
from judge import check_numbers, evaluate_relevance
from metrics import RAGWithMetrics
from rag_helper import INSTRUCTIONS

BASIC_INSTRUCTIONS = """
Your task is to answer questions about the Cambodian budget based on the provided context.
If the answer is not found in the context, respond with "I don't know."
""".strip()

VARIANTS = {
    "basic prompt (course project)": BASIC_INSTRUCTIONS,
    "budget prompt (used)": INSTRUCTIONS,
}


def patient(call, what):
    """Run an LLM call; on a Groq rate limit, wait and try again instead of stopping."""
    for attempt in range(20):
        try:
            return call()
        except RateLimitError as e:
            if "Request too large" in str(e):
                raise SystemExit("This model's per-minute limit is smaller than one request. "
                                 "Use openai/gpt-oss-120b or lower ANSWER_MAX_TOKENS / JUDGE_MAX_TOKENS.")
            if "per day" in str(e) or "TPD" in str(e) or "RPD" in str(e):
                raise SystemExit("Groq daily limit reached. Results so far are lost; run again tomorrow "
                                 "with a smaller --limit.")
            try:
                wait = float(e.response.headers.get("retry-after") or 30)
            except Exception:
                wait = 30
            print(f"   rate limit on {what}, waiting {wait:.0f}s ...", flush=True)
            time.sleep(wait + 1)
    raise SystemExit("Still rate-limited after 20 tries; try again later.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    load_dotenv()
    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    index = build_index(load_budget_data())

    with open(BASE_DIR / "data" / "test_questions.csv", encoding="utf-8") as f:
        questions = [r["question"] for r in csv.DictReader(f)][: args.limit]

    rows = []
    for name, instructions in VARIANTS.items():
        rag = RAGWithMetrics(index=index, llm_client=client, model=model, instructions=instructions)
        for i, q in enumerate(questions, 1):
            answer = patient(lambda: rag.rag(q), "answer")
            rec = rag.last_call
            num_status, missing = check_numbers(answer, rec.prompt)
            verdict, explanation = patient(
                lambda: evaluate_relevance(q, answer, context=rec.prompt, client=client, model=model), "judge")
            rows.append({"variant": name, "question": q, "answer": answer, "judge": verdict,
                         "explanation": explanation, "number_check": num_status,
                         "missing_numbers": ", ".join(missing), "response_time": round(rec.response_time, 2),
                         "cost": rec.cost})
            print(f"[{name}] {i}/{len(questions)} {verdict:15s} {num_status:10s} {q[:50]}", flush=True)
            time.sleep(0.5)  # stay under free-plan rate limits

    out = Path(BASE_DIR / "evaluation"); out.mkdir(exist_ok=True)
    with open(out / "llm_results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f"\nModel: {model}, {len(questions)} questions\n")
    print("| Prompt | CORRECT | PARTLY | INCORRECT | Numbers match docs | Avg time (s) | Cost ($) |")
    print("|---|---|---|---|---|---|---|")
    for name in VARIANTS:
        r = [x for x in rows if x["variant"] == name]
        n = len(r)
        pct = lambda key, val: f"{100 * sum(x[key] == val for x in r) / n:.0f}%"
        with_numbers = [x for x in r if x["number_check"] != "NO_NUMBERS"]
        ok = (f"{100 * sum(x['number_check'] == 'OK' for x in with_numbers) / len(with_numbers):.0f}%"
              if with_numbers else "-")
        print(f"| {name} | {pct('judge', 'CORRECT')} | {pct('judge', 'PARTLY_CORRECT')} | "
              f"{pct('judge', 'INCORRECT')} | {ok} | {sum(x['response_time'] for x in r) / n:.2f} | "
              f"{sum(x['cost'] for x in r):.4f} |")
    print(f"\nDetails: {out / 'llm_results.csv'}")


if __name__ == "__main__":
    main()