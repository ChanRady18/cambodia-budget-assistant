"""
Turn structured budget tables into RAG documents (Q&A pairs).

Input  (data/):
  budget_lines.csv  one row per (year, BU, type, metric) -- the numbers you copy
                    out of the budget law / settlement law PDFs
  macro.csv         GDP and exchange rate per year
  aliases.csv       optional nicknames / abbreviations per BU (helps search)

Output (documents/):
  all_documents.json   the knowledge base the assistant searches
  ground_truth.json    test questions -> expected doc id (for retrieval eval)

Every number in an answer is computed here, in code, never typed by hand.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "documents"

KH_DIGITS = str.maketrans("0123456789", "០១២៣៤៥៦៧៨៩")

TYPE_KH = {"budget": "ថវិកាគ្រោង", "settlement": "ការអនុវត្តជាក់ស្តែង"}
METRIC = {
    # (type, metric): (english phrase, khmer phrase)
    ("budget", "expenditure"): ("planned expenditure", "ថវិកាគ្រោងចំណាយ"),
    ("budget", "revenue"): ("planned revenue", "ចំណូលគ្រោងប្រមូល"),
    ("settlement", "expenditure"): ("actual expenditure", "ចំណាយជាក់ស្តែង"),
    ("settlement", "revenue"): ("actual revenue collected", "ចំណូលប្រមូលបានជាក់ស្តែង"),
}


# ---------------------------------------------------------------- helpers

def kh(n):
    """Western digits -> Khmer digits (for years and percentages in Khmer text)."""
    return str(n).translate(KH_DIGITS)


def riel(n):
    return f"{n:,.0f} riel"


def riel_kh(n):
    return f"{n:,.0f} រៀល"


def pct(x):
    return f"{x:.2f}%"


def usd(n, rate):
    if not rate:
        return ""
    v = n / rate
    return f" (≈ {v / 1e9:,.2f} billion USD)" if v >= 1e9 else f" (≈ {v / 1e6:,.1f} million USD)"


def ref(*rows):
    """Citation string from one or more source rows."""
    parts = []
    for r in rows:
        if r and r.get("file_ref"):
            parts.append(f"{r['file_ref']} ({r['file_section']})")
    return "; ".join(dict.fromkeys(parts))


def make_doc(doc_id, year, bu, type_, question, question_kh, answer, refs, **extra):
    doc = {
        "id": doc_id,
        "section": "budget",
        "budget_year": str(year),
        "bu_code": bu["bu_code"],
        "BU": bu["bu_kh"],
        "bu_en": bu["bu_en"],
        "type": type_,
        "question": question,
        "question_kh": question_kh,
        "answer": answer,
        "keywords": bu.get("aliases", ""),
        "file_ref": "; ".join(dict.fromkeys(r["file_ref"] for r in refs if r)),
        "section_file_ref": "; ".join(
            dict.fromkeys(f"{r['file_ref']}#{r['file_section']}" for r in refs if r)
        ),
    }
    doc.update(extra)
    return doc


# ---------------------------------------------------------------- loading + validation

def load():
    errors = []
    lines = {}
    bus = {}

    aliases = {}
    if (DATA / "aliases.csv").exists():
        with open(DATA / "aliases.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                aliases[r["bu_code"].strip()] = r["aliases"].strip()

    with open(DATA / "budget_lines.csv", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f), start=2):
            r = {k: (v or "").strip() for k, v in r.items()}
            key = (int(r["budget_year"]), r["bu_code"], r["type"], r["metric"])
            if (r["type"], r["metric"]) not in METRIC:
                errors.append(f"line {i}: unknown type/metric {r['type']}/{r['metric']}")
                continue
            try:
                r["amount_riel"] = int(r["amount_riel"].replace(",", ""))
            except ValueError:
                errors.append(f"line {i}: amount_riel is not a whole number: {r['amount_riel']!r}")
                continue
            if r["amount_riel"] < 0:
                errors.append(f"line {i}: negative amount")
            if key in lines:
                errors.append(f"line {i}: duplicate row for {key}")
            lines[key] = r
            bus.setdefault(r["bu_code"], {
                "bu_code": r["bu_code"], "bu_kh": r["bu_kh"], "bu_en": r["bu_en"],
                "aliases": aliases.get(r["bu_code"], ""),
            })

    macro = {}
    if (DATA / "macro.csv").exists():
        with open(DATA / "macro.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r = {k: (v or "").strip() for k, v in r.items()}
                macro[int(r["year"])] = {
                    "gdp_riel": int(r["gdp_riel"]) if r["gdp_riel"] else None,
                    "khr_per_usd": float(r["khr_per_usd"]) if r["khr_per_usd"] else None,
                    "file_ref": r.get("file_ref", ""),
                    "file_section": r.get("file_section", ""),
                }

    if errors:
        print("Input errors:\n  " + "\n  ".join(errors), file=sys.stderr)
        sys.exit(1)
    return lines, bus, macro


# ---------------------------------------------------------------- document builders

def fact_docs(lines, bus, macro):
    """One document per number: e.g. MPWT planned expenditure 2024."""
    docs, gt = [], []
    for (year, code, type_, metric), r in sorted(lines.items()):
        bu = bus[code]
        en, khp = METRIC[(type_, metric)]
        amt = r["amount_riel"]
        rate = macro.get(year, {}).get("khr_per_usd")
        law_en = "budget law" if type_ == "budget" else "budget settlement law"

        q = f"What is the {en} of {bu['bu_en']} in {year}?"
        qk = f"តើ{bu['bu_kh']}មាន{khp}ប៉ុន្មាន សម្រាប់ឆ្នាំ{kh(year)}?"
        a = (
            f"{khp}របស់{bu['bu_kh']} សម្រាប់ឆ្នាំ{kh(year)} គឺ {riel_kh(amt)}។\n"
            f"{bu['bu_en']} {en} for {year}: {riel(amt)}{usd(amt, rate)}, "
            f"according to the {year} {law_en}.\n"
            f"Source: {ref(r)}"
        )
        doc_id = f"{year}-{code}-{type_}-{metric}".lower()
        docs.append(make_doc(doc_id, year, bu, type_, q, qk, a, [r],
                             metric=metric, amount_riel=amt))
        gt += [{"question": q, "doc_id": doc_id}, {"question": qk, "doc_id": doc_id}]
    return docs, gt


def execution_docs(lines, bus, macro):
    """Planned vs actual, with execution rate -- the 'settlement' comparison."""
    docs, gt = [], []
    keys = {(y, c) for (y, c, _, _) in lines}
    for year, code in sorted(keys):
        bu = bus[code]
        parts_kh, parts_en, refs, extra = [], [], [], {}
        for metric, word_en, word_kh in [
            ("expenditure", "spent", "ចំណាយ"),
            ("revenue", "collected", "ប្រមូលចំណូល"),
        ]:
            plan = lines.get((year, code, "budget", metric))
            act = lines.get((year, code, "settlement", metric))
            if not (plan and act):
                continue
            p, a = plan["amount_riel"], act["amount_riel"]
            rate_ = a / p * 100 if p else 0
            parts_kh.append(
                f"{bu['bu_kh']}បាន{word_kh}ជាក់ស្តែង {riel_kh(a)} ធៀបនឹងច្បាប់ថវិកាដែលបានគ្រោង "
                f"{riel_kh(p)} ស្មើនឹងអត្រាអនុវត្ត {kh(pct(rate_))}។"
            )
            parts_en.append(
                f"{metric.capitalize()}: {word_en} {riel(a)} against {riel(p)} planned "
                f"= {pct(rate_)} execution ({'+' if a >= p else '-'}{riel(abs(a - p))})."
            )
            refs += [plan, act]
            extra[f"{metric}_planned_riel"] = p
            extra[f"{metric}_actual_riel"] = a
            extra[f"{metric}_execution_rate"] = round(rate_, 2)
        if not parts_en:
            continue
        q = f"How much did {bu['bu_en']} actually spend in {year} compared to the budget (execution rate)?"
        qk = f"តើ{bu['bu_kh']}អនុវត្តថវិកាជាក់ស្តែងបានប៉ុន្មាន ធៀបនឹងថវិកាគ្រោង ក្នុងឆ្នាំ{kh(year)}?"
        a = "\n".join(parts_kh + parts_en) + f"\nSource: {ref(*refs)}"
        doc_id = f"{year}-{code}-execution".lower()
        docs.append(make_doc(doc_id, year, bu, "execution", q, qk, a, refs, **extra))
        gt += [{"question": q, "doc_id": doc_id}, {"question": qk, "doc_id": doc_id},
               {"question": f"{bu['bu_en']} budget execution rate {year}", "doc_id": doc_id}]
    return docs, gt


def growth_docs(lines, bus, macro):
    """Year-over-year change in planned expenditure."""
    docs, gt = [], []
    for (year, code, type_, metric), r in sorted(lines.items()):
        if type_ != "budget" or metric != "expenditure":
            continue
        prev = lines.get((year - 1, code, "budget", "expenditure"))
        if not prev:
            continue
        bu = bus[code]
        a, b = r["amount_riel"], prev["amount_riel"]
        g = (a - b) / b * 100
        direction_kh = "កើនឡើង" if g >= 0 else "ថយចុះ"
        q = f"How did the budget of {bu['bu_en']} change from {year - 1} to {year}?"
        qk = f"តើថវិការបស់{bu['bu_kh']} ឆ្នាំ{kh(year)} ប្រែប្រួលយ៉ាងដូចម្តេច ធៀបនឹងឆ្នាំ{kh(year - 1)}?"
        ans = (
            f"ថវិកាគ្រោងចំណាយរបស់{bu['bu_kh']} {direction_kh} {kh(pct(abs(g)))} "
            f"ពី {riel_kh(b)} (ឆ្នាំ{kh(year - 1)}) ទៅ {riel_kh(a)} (ឆ្នាំ{kh(year)})។\n"
            f"{bu['bu_en']} planned expenditure {'grew' if g >= 0 else 'fell'} {pct(abs(g))} "
            f"from {riel(b)} in {year - 1} to {riel(a)} in {year} ({'+' if g >= 0 else '-'}{riel(abs(a - b))}).\n"
            f"Source: {ref(prev, r)}"
        )
        doc_id = f"{year}-{code}-growth".lower()
        docs.append(make_doc(doc_id, year, bu, "growth", q, qk, ans, [prev, r],
                             growth_pct=round(g, 2)))
        gt += [{"question": q, "doc_id": doc_id}, {"question": qk, "doc_id": doc_id}]
    return docs, gt


def gdp_docs(lines, bus, macro):
    """Share of state budget and % of GDP, per BU and for the state total."""
    docs, gt = [], []
    for (year, code, type_, metric), r in sorted(lines.items()):
        if type_ != "budget" or metric != "expenditure":
            continue
        m = macro.get(year)
        state = lines.get((year, "STATE", "budget", "expenditure"))
        if not m or not m["gdp_riel"]:
            continue
        bu = bus[code]
        amt, gdp = r["amount_riel"], m["gdp_riel"]
        of_gdp = amt / gdp * 100
        gdp_row = {"file_ref": m["file_ref"], "file_section": m["file_section"]}
        kh_lines = [f"ថវិកាគ្រោងចំណាយរបស់{bu['bu_kh']} ឆ្នាំ{kh(year)} ({riel_kh(amt)}) "
                    f"ស្មើនឹង {kh(pct(of_gdp))} នៃផលិតផលក្នុងស្រុកសរុប (ផ.ស.ស) {riel_kh(gdp)}។"]
        en_lines = [f"{bu['bu_en']} planned expenditure {year}: {riel(amt)} = {pct(of_gdp)} of GDP "
                    f"- GDP {year}: {riel(gdp)}{usd(gdp, m['khr_per_usd'])}."]
        extra = {"pct_of_gdp": round(of_gdp, 3), "gdp_riel": gdp}
        refs = [r, gdp_row]
        if state and code != "STATE":
            share = amt / state["amount_riel"] * 100
            kh_lines.append(f"ស្មើនឹង {kh(pct(share))} នៃថវិកាចំណាយសរុបរបស់រដ្ឋ {riel_kh(state['amount_riel'])}។")
            en_lines.append(f"That is {pct(share)} of total state expenditure ({riel(state['amount_riel'])}).")
            extra["pct_of_state_budget"] = round(share, 2)
            refs.insert(1, state)
        q = f"What share of GDP is the {year} budget of {bu['bu_en']}?"
        qk = f"តើថវិការបស់{bu['bu_kh']} ឆ្នាំ{kh(year)} ស្មើនឹងប៉ុន្មានភាគរយនៃ ផ.ស.ស?"
        ans = "\n".join(kh_lines + en_lines) + f"\nSource: {ref(*refs)}"
        doc_id = f"{year}-{code}-gdp".lower()
        docs.append(make_doc(doc_id, year, bu, "gdp", q, qk, ans, refs, **extra))
        gt += [{"question": q, "doc_id": doc_id}, {"question": qk, "doc_id": doc_id},
               {"question": f"{bu['bu_en']} budget as percentage of GDP {year}", "doc_id": doc_id}]
    return docs, gt


def macro_docs(lines, bus, macro):
    docs, gt = [], []
    gdp_bu = {"bu_code": "MACRO", "bu_kh": "ម៉ាក្រូសេដ្ឋកិច្ច", "bu_en": "Macroeconomy",
              "aliases": "GDP; gross domestic product; ផ.ស.ស; ផលិតផលក្នុងស្រុកសរុប; exchange rate; អត្រាប្តូរប្រាក់"}
    for year, m in sorted(macro.items()):
        if not m["gdp_riel"]:
            continue
        row = {"file_ref": m["file_ref"], "file_section": m["file_section"]}
        rate_line = (f"\nExchange rate assumption: {m['khr_per_usd']:,.0f} riel per USD."
                     if m["khr_per_usd"] else "")
        q = f"What is Cambodia's GDP in {year}?"
        qk = f"តើផលិតផលក្នុងស្រុកសរុប (ផ.ស.ស) ឆ្នាំ{kh(year)} មានប៉ុន្មាន?"
        ans = (f"ផ.ស.ស ឆ្នាំ{kh(year)} គឺ {riel_kh(m['gdp_riel'])}។\n"
               f"GDP {year}: {riel(m['gdp_riel'])}{usd(m['gdp_riel'], m['khr_per_usd'])}.{rate_line}\n"
               f"Source: {ref(row)}")
        doc_id = f"{year}-macro-gdp"
        docs.append(make_doc(doc_id, year, gdp_bu, "macro", q, qk, ans, [row],
                             gdp_riel=m["gdp_riel"], khr_per_usd=m["khr_per_usd"]))
        gt += [{"question": q, "doc_id": doc_id}, {"question": qk, "doc_id": doc_id}]
    return docs, gt


def ranking_docs(lines, bus, macro):
    """Per-year rankings, so 'which ministry has the biggest budget' is answerable."""
    docs, gt = [], []
    years = sorted({y for (y, *_rest) in lines})
    summary_bu = {"bu_code": "ALL", "bu_kh": "គ្រប់ក្រសួង-ស្ថាប័ន", "bu_en": "All ministries and institutions",
                  "aliases": "ranking; largest; biggest; highest; lowest; top; compare ministries; ចំណាត់ថ្នាក់; ធំជាងគេ"}
    for year in years:
        rows = [r for (y, c, t, m), r in lines.items()
                if y == year and c != "STATE" and t == "budget" and m == "expenditure"]
        if rows:
            rows.sort(key=lambda r: -r["amount_riel"])
            listing = "\n".join(f"{i}. {r['bu_kh']} / {r['bu_en']}: {riel(r['amount_riel'])}"
                                for i, r in enumerate(rows, 1))
            q = f"Which ministries have the largest budget in {year}?"
            qk = f"តើក្រសួង-ស្ថាប័នណាខ្លះមានថវិកាគ្រោងចំណាយច្រើនជាងគេ ក្នុងឆ្នាំ{kh(year)}?"
            ans = (f"ចំណាត់ថ្នាក់ថវិកាគ្រោងចំណាយ ឆ្នាំ{kh(year)} / Planned expenditure ranking {year} "
                   f"(ministries in this dataset):\n{listing}\nSource: {ref(*rows)}")
            doc_id = f"{year}-all-ranking-budget"
            docs.append(make_doc(doc_id, year, summary_bu, "summary", q, qk, ans, rows))
            gt += [{"question": q, "doc_id": doc_id}, {"question": qk, "doc_id": doc_id}]

        exec_rows = []
        for code in bus:
            p = lines.get((year, code, "budget", "expenditure"))
            a = lines.get((year, code, "settlement", "expenditure"))
            if code != "STATE" and p and a and p["amount_riel"]:
                exec_rows.append((a["amount_riel"] / p["amount_riel"] * 100, p, a))
        if exec_rows:
            exec_rows.sort(key=lambda x: -x[0])
            listing = "\n".join(f"{i}. {p['bu_kh']} / {p['bu_en']}: {pct(rt)} "
                                f"({riel(a['amount_riel'])} of {riel(p['amount_riel'])})"
                                for i, (rt, p, a) in enumerate(exec_rows, 1))
            q = f"Which ministries had the highest and lowest budget execution rate in {year}?"
            qk = f"តើក្រសួង-ស្ថាប័នណាអនុវត្តថវិកាបានខ្ពស់ជាងគេ និងទាបជាងគេ ក្នុងឆ្នាំ{kh(year)}?"
            ans = (f"អត្រាអនុវត្តថវិកាចំណាយ ឆ្នាំ{kh(year)} / Expenditure execution rate {year}:\n{listing}\n"
                   f"Source: {ref(*[x for _, p, a in exec_rows for x in (p, a)])}")
            doc_id = f"{year}-all-ranking-execution"
            docs.append(make_doc(doc_id, year, summary_bu, "summary", q, qk, ans,
                                 [x for _, p, a in exec_rows for x in (p, a)]))
            gt += [{"question": q, "doc_id": doc_id}, {"question": qk, "doc_id": doc_id}]
    return docs, gt


BUILDERS = [fact_docs, execution_docs, growth_docs, gdp_docs, macro_docs, ranking_docs]


def main():
    lines, bus, macro = load()
    docs, gt = [], []
    for build in BUILDERS:
        d, g = build(lines, bus, macro)
        docs += d
        gt += g

    ids = [d["id"] for d in docs]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate ids: {dupes}"

    OUT.mkdir(exist_ok=True)
    with open(OUT / "all_documents.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    with open(OUT / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)

    counts = defaultdict(int)
    for d in docs:
        counts[d["type"]] += 1
    print(f"{len(docs)} documents -> documents/all_documents.json  {dict(counts)}")
    print(f"{len(gt)} test questions -> documents/ground_truth.json")


if __name__ == "__main__":
    main()
