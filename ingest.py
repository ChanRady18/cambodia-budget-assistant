import json
from pathlib import Path

from minsearch import Index

BASE_DIR = Path(__file__).resolve().parent

TEXT_FIELDS = ["question", "question_kh", "answer", "BU", "bu_en", "keywords"]
KEYWORD_FIELDS = ["section", "budget_year", "bu_code", "type"]

# Khmer has no spaces between words and its vowel signs are not "word" characters
# for the default regex tokenizer, so "ក្រសួងសាធារណការ" gets shredded into junk.
# Character n-grams work for Khmer and English without a word segmenter.
VECTORIZER_PARAMS = {"analyzer": "char_wb", "ngram_range": (2, 4), "token_pattern": None}


def load_budget_data(file_path=None):
    if file_path is None:
        file_path = BASE_DIR / "documents" / "all_documents.json"

    with open(file_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    for doc in documents:
        doc["doc_id"] = doc.pop("id")

    return documents


def build_index(documents):
    index = Index(
        text_fields=TEXT_FIELDS,
        keyword_fields=KEYWORD_FIELDS,
        vectorizer_params=VECTORIZER_PARAMS,
    )
    index.fit(documents)
    return index
