"""
Compare retrieval strategies on the same test questions.

  python evaluate_retrieval_strategies.py          -> prints a Markdown table

Test sets
  hand-written : data/test_questions.csv (40 realistic Khmer / English / mixed questions)
  generated    : documents/ground_truth.json (the template questions, 116; easy, catches regressions)

Metrics (top 5 results)
  hit rate = share of questions whose correct document is in the top 5
  MRR      = mean of 1 / rank of the correct document (0 if not in the top 5)

Optional: if `sentence-transformers` is installed, a multilingual embedding model and a
hybrid with it are evaluated too (needs internet the first time to download the model).
"""
import csv
import json

import numpy as np
from minsearch import Index
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from ingest import BASE_DIR, KEYWORD_FIELDS, TEXT_FIELDS, load_budget_data
from rag_helper import detect_year

K = 5
CURRENT_BOOST = {"question": 3.0, "question_kh": 3.0, "BU": 2.0, "bu_en": 2.0, "keywords": 1.5, "answer": 0.5}
CHAR = lambda lo, hi: {"analyzer": "char_wb", "ngram_range": (lo, hi), "token_pattern": None}


def load_tests():
    with open(BASE_DIR / "data" / "test_questions.csv", encoding="utf-8") as f:
        hand = [{"question": r["question"], "ids": r["doc_id"].split("|")} for r in csv.DictReader(f)]
    with open(BASE_DIR / "documents" / "ground_truth.json", encoding="utf-8") as f:
        gen = [{"question": r["question"], "ids": [r["doc_id"]]} for r in json.load(f)]
    return hand, gen


def score(search, tests):
    hits, rr = 0, 0.0
    for t in tests:
        ids = search(t["question"])[:K]
        ranks = [ids.index(w) + 1 for w in t["ids"] if w in ids]
        if ranks:
            hits += 1
            rr += 1 / min(ranks)
    return hits / len(tests), rr / len(tests)


# ---------------------------------------------------------------- text search (minsearch TF-IDF)

def text_search(docs, text_fields, vectorizer_params=None, boost=None, year_filter=True):
    idx = Index(text_fields=text_fields, keyword_fields=KEYWORD_FIELDS, vectorizer_params=vectorizer_params)
    idx.fit(docs)

    def search(q):
        flt = {"section": "budget"}
        y = detect_year(q) if year_filter else None
        if y:
            flt["budget_year"] = y
        res = idx.search(q, filter_dict=flt, boost_dict=boost or {}, num_results=K * 4)
        return [d["doc_id"] for d in res]
    return search


# ---------------------------------------------------------------- vector search

def doc_text(d):
    return " ".join(str(d.get(f, "")) for f in ["question", "question_kh", "BU", "bu_en", "keywords", "answer"])


def vector_search_from_matrix(docs, doc_vecs, embed, year_filter=True):
    doc_vecs = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9)
    years = np.array([d["budget_year"] for d in docs])
    ids = [d["doc_id"] for d in docs]

    def search(q):
        v = embed(q)
        v = v / (np.linalg.norm(v) + 1e-9)
        sims = doc_vecs @ v
        y = detect_year(q) if year_filter else None
        if y:
            sims = np.where(years == y, sims, -1)
        return [ids[i] for i in np.argsort(-sims)[: K * 4]]
    return search


def lsa_search(docs, dims=100):
    """'Embeddings' without a model download: char n-gram TF-IDF reduced with SVD (LSA)."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    X = vec.fit_transform([doc_text(d) for d in docs])
    svd = TruncatedSVD(n_components=min(dims, X.shape[1] - 1, len(docs) - 1), random_state=42)
    D = svd.fit_transform(X)
    return vector_search_from_matrix(docs, D, lambda q: svd.transform(vec.transform([q]))[0])


def neural_search(docs, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    from sentence_transformers import SentenceTransformer  # optional dependency
    model = SentenceTransformer(model_name)
    D = model.encode([doc_text(d) for d in docs], normalize_embeddings=True)
    return vector_search_from_matrix(docs, D, lambda q: model.encode([q], normalize_embeddings=True)[0])


# ---------------------------------------------------------------- hybrid

def hybrid(*searches, k=60):
    """Reciprocal Rank Fusion: score = sum of 1 / (k + rank) over the result lists."""
    def search(q):
        scores = {}
        for s in searches:
            for rank, doc_id in enumerate(s(q), start=1):
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
        return sorted(scores, key=scores.get, reverse=True)
    return search


def main():
    docs = load_budget_data()
    hand, gen = load_tests()

    current = text_search(docs, TEXT_FIELDS, CHAR(2, 4), CURRENT_BOOST)
    lsa = lsa_search(docs)
    strategies = [
        ("Text, word tokens, course-project setup (question/section/answer, no year filter)",
         text_search(docs, ["question", "section", "answer"], None, {"question": 3.0, "section": 0.5}, year_filter=False)),
        ("Text, word tokens, all budget fields + year filter",
         text_search(docs, TEXT_FIELDS, None, CURRENT_BOOST)),
        ("Text, char 2-4 n-grams, no year filter",
         text_search(docs, TEXT_FIELDS, CHAR(2, 4), CURRENT_BOOST, year_filter=False)),
        ("Text, char 2-4 n-grams + year filter, no boosts",
         text_search(docs, TEXT_FIELDS, CHAR(2, 4), None)),
        ("Text, char 1-3 n-grams + year filter",
         text_search(docs, TEXT_FIELDS, CHAR(1, 3), CURRENT_BOOST)),
        ("Text, char 2-5 n-grams + year filter",
         text_search(docs, TEXT_FIELDS, CHAR(2, 5), CURRENT_BOOST)),
        ("**Text, char 2-4 n-grams + year filter + boosts (used)**", current),
        ("Vector, LSA (char TF-IDF + SVD, 100 dims) + year filter", lsa),
        ("Hybrid RRF: used text search + LSA vector", hybrid(current, lsa)),
    ]
    try:
        neural = neural_search(docs)
        strategies += [
            ("Vector, multilingual MiniLM embeddings + year filter", neural),
            ("Hybrid RRF: used text search + MiniLM", hybrid(current, neural)),
        ]
    except Exception as e:  # not installed / no internet
        print(f"(neural embeddings skipped: {type(e).__name__})\n")

    print(f"| Strategy | Hand-written ({len(hand)}) hit@5 | MRR | Generated ({len(gen)}) hit@5 | MRR |")
    print("|---|---|---|---|---|")
    for name, s in strategies:
        h_hit, h_mrr = score(s, hand)
        g_hit, g_mrr = score(s, gen)
        print(f"| {name} | {h_hit:.3f} | {h_mrr:.3f} | {g_hit:.3f} | {g_mrr:.3f} |")


if __name__ == "__main__":
    main()
