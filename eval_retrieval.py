"""Retrieval check without an LLM: does the right document come back for each test question?"""
import csv
import json
import sys

from ingest import BASE_DIR, build_index, load_budget_data
from rag_helper import RAGBase


def evaluate(index):
    rag = RAGBase(index=index, llm_client=None)
    with open(BASE_DIR / "documents" / "ground_truth.json", encoding="utf-8") as f:
        gt = json.load(f)
    # hand-written paraphrases are the real test; generated ones only catch regressions
    hand = BASE_DIR / "data" / "test_questions.csv"
    if hand.exists():
        with open(hand, encoding="utf-8") as f:
            gt += list(csv.DictReader(f))
    hits, rr, misses = 0, 0.0, []
    for item in gt:
        ids = [d["doc_id"] for d in rag.search(item["question"], num_results=5)]
        wanted = item["doc_id"].split("|")          # "a|b" = either document is a correct answer
        ranks = [ids.index(w) + 1 for w in wanted if w in ids]
        if ranks:
            hits += 1
            rr += 1 / min(ranks)
        else:
            misses.append((item["question"], item["doc_id"], ids[:3]))
    n = len(gt)
    return hits / n, rr / n, misses


if __name__ == "__main__":
    docs = load_budget_data()
    hit_rate, mrr, misses = evaluate(build_index(docs))
    print(f"hit rate@5: {hit_rate:.3f}   MRR: {mrr:.3f}   ({len(misses)} misses)")
    if "-v" in sys.argv:
        for q, want, got in misses:
            print(f"- {q}\n    want {want}, got {got}")
