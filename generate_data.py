import time
import random

from metrics import LLMCallRecord
from db_save import save_conversation
from db_feedback import save_feedback
from db_sessions import create_session, touch_session


SAMPLE_QUESTIONS = [
    "How much is the 2024 budget of the Ministry of Health?",
    "តើក្រសួងសាធារណការ និងដឹកជញ្ជូនមានថវិកាប៉ុន្មាន ឆ្នាំ២០២៤?",
    "What was the execution rate of MoEYS in 2023?",
    "What share of GDP is the education budget in 2024?",
    "Which ministry has the largest budget in 2024?",
]

SAMPLE_ANSWERS = [
    "The Ministry of Health planned expenditure for 2024 is 3,350,000,000,000 riel.",
    "ថវិកាគ្រោងចំណាយរបស់ក្រសួងសាធារណការ និងដឹកជញ្ជូន ឆ្នាំ២០២៤ គឺ 1,520,000,000,000 រៀល។",
    "MoEYS spent 4,210,000,000,000 riel of 4,300,000,000,000 planned = 97.91%.",
    "The education budget is 2.54% of GDP in 2024.",
    "The Ministry of Education, Youth and Sport has the largest budget in 2024.",
]

RELEVANCE = ["CORRECT", "CORRECT", "CORRECT", "PARTLY_CORRECT", "INCORRECT"]
NUMBER_STATUS = ["OK", "OK", "OK", "MISMATCH", "NO_NUMBERS"]


def fake_record(question, answer):
    return LLMCallRecord(
        model="openai/gpt-oss-120b",
        prompt=question,
        instructions="",
        answer=answer,
        prompt_tokens=random.randint(50, 200),
        completion_tokens=random.randint(50, 300),
        total_tokens=random.randint(100, 500),
        response_time=random.uniform(0.5, 5.0),
        cost=random.uniform(0.0001, 0.01),
    )


def random_score():
    return random.choice([1, 1, 1, 1, -1])


_current = {"session": None, "left": 0}


def _session():
    """Group fake questions into chats of 1-4 questions."""
    if _current["left"] <= 0:
        _current["session"] = create_session()
        _current["left"] = random.randint(1, 4)
        _current["new"] = True
    _current["left"] -= 1
    return _current["session"]


def generate_one():
    session_id = _session()
    question = random.choice(SAMPLE_QUESTIONS)
    answer = random.choice(SAMPLE_ANSWERS)
    record = fake_record(question, answer)

    conversation_id = save_conversation(
        record, question, random.choice(["2023", "2024", "2024", "unknown"]), session_id=session_id
    )
    touch_session(session_id, first_question=question if _current.pop("new", False) else None)

    save_feedback(conversation_id, "number_check", relevance=random.choice(NUMBER_STATUS))

    if random.random() < 0.7:
        relevance = random.choice(RELEVANCE)
        save_feedback(
            conversation_id, "judge",
            relevance=relevance,
            explanation=f"Answer is {relevance.lower()}.",
        )

    if random.random() < 0.5:
        score = random_score()
        save_feedback(conversation_id, "user", score=score)


def generate_live():
    print("Starting live data generation (Ctrl+C to stop)...", flush=True)
    while True:
        generate_one()
        time.sleep(1)


if __name__ == "__main__":
    try:
        generate_live()
    except KeyboardInterrupt:
        print("Stopped.")
