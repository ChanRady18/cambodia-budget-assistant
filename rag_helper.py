import os
import re


def answer_max_tokens():
    """Cap on the answer's length. Groq counts this cap against per-minute token limits."""
    return int(os.getenv("ANSWER_MAX_TOKENS", "1024"))


INSTRUCTIONS = '''
You are an assistant for questions about the Cambodian state budget
(budget laws, budget settlement laws and macroeconomic data).

Rules:
- Answer ONLY from the CONTEXT. Copy numbers exactly as written; do not round
  or re-calculate them unless the user asks for a calculation.
- Always state the year, the ministry/institution and the unit (riel).
- Say whether a figure is planned (budget law) or actual (settlement law).
- End with the source file and section shown in the context.
- Reply in the language of the question (Khmer or English).
- If the context does not contain the answer, say "I don't know" /
  "ខ្ញុំមិនមានព័ត៌មាននេះទេ".
- EARLIER IN THIS CHAT (if present) is only there to understand follow-up
  questions such as "and in 2023?". Take every number from CONTEXT, never from it.
'''

PROMPT_TEMPLATE = '''
{history}QUESTION: {question}

CONTEXT:
{context}
'''.strip()

HISTORY_TURNS = 3          # how many earlier question/answer pairs the model sees
FOLLOW_UP_MAX_CHARS = 25   # a question shorter than this is treated as a follow-up

KH_TO_EN_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")
YEAR_RE = re.compile(r"\b(20\d\d)\b")


def detect_year(query):
    """Return the year if the question mentions exactly one (Khmer or Western digits)."""
    years = set(YEAR_RE.findall(query.translate(KH_TO_EN_DIGITS)))
    return years.pop() if len(years) == 1 else None


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        budget_year=None,
        model='gpt-5.4-mini'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.budget_year = budget_year   # None = auto-detect from the question
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5, year=None):
        boost_dict = {"question": 3.0, "question_kh": 3.0, "BU": 2.0,
                      "bu_en": 2.0, "keywords": 1.5, "answer": 0.5}
        filter_dict = {"section": "budget"}

        year = self.budget_year or year or detect_year(query)
        if year:
            filter_dict["budget_year"] = str(year)

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def build_context(self, search_results):
        lines = []
        for doc in search_results:
            lines.append(f"[{doc['budget_year']} | {doc['BU']} / {doc['bu_en']} | {doc['type']}]")
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")
        return "\n".join(lines).strip()

    def build_history(self, history):
        if not history:
            return ""
        lines = ["EARLIER IN THIS CHAT:"]
        for turn in history[-HISTORY_TURNS:]:
            lines.append("User: " + turn["question"])
            lines.append("Assistant: " + turn["answer"][:600])
        return "\n".join(lines) + "\n\n"

    def build_prompt(self, query, search_results, history=None):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            history=self.build_history(history), question=query, context=context
        )

    def search_for(self, query, history=None):
        """
        Search, taking earlier turns into account:
        - no year in the question -> reuse the year of the previous question
        - a very short question ("and MoH?", "ចុះឆ្នាំ២០២៣?") -> search it together with the previous one
        """
        prev = history[-1]["question"] if history else ""
        year = detect_year(query) or (detect_year(prev) if prev else None)
        search_text = query
        if prev and len(query.strip()) < FOLLOW_UP_MAX_CHARS:
            search_text = prev + " " + query
        return self.search(search_text, year=year)

    def llm(self, prompt):
        input_messages = [
            {'role': 'system', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]
        response = self.llm_client.responses.create(model=self.model, input=input_messages,
                                                    max_output_tokens=answer_max_tokens())
        return response.output_text

    def rag(self, query, history=None):
        """history = earlier turns of this chat: [{"question": ..., "answer": ...}, ...]"""
        search_results = self.search_for(query, history)
        prompt = self.build_prompt(query, search_results, history)
        return self.llm(prompt)
