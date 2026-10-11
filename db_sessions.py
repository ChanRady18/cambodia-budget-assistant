"""Chat sessions: each chat has a UUID; its messages are rows in `conversations`."""
import uuid
from datetime import datetime

from db_init import DB_TIMEZONE, get_db_connection

TITLE_LENGTH = 60


def create_session():
    """Make a new chat and return its UUID (as a string)."""
    session_id = uuid.uuid4()
    now = datetime.now(DB_TIMEZONE)
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat_sessions (id, title, created_at, updated_at) VALUES (%s, %s, %s, %s)",
            (session_id, "New chat", now, now),
        )
    return str(session_id)


def session_exists(session_id):
    try:
        uuid.UUID(str(session_id))
    except ValueError:
        return False
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM chat_sessions WHERE id = %s", (session_id,))
        return cur.fetchone() is not None


def touch_session(session_id, first_question=None):
    """Update the 'last active' time; name the chat after its first question."""
    now = datetime.now(DB_TIMEZONE)
    title = None
    if first_question:
        q = " ".join(first_question.split())
        title = q if len(q) <= TITLE_LENGTH else q[:TITLE_LENGTH - 1] + "…"
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """UPDATE chat_sessions
               SET updated_at = %s,
                   title = CASE WHEN title = 'New chat' AND %s::text IS NOT NULL THEN %s ELSE title END
               WHERE id = %s""",
            (now, title, title, session_id),
        )


def rename_session(session_id, title):
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE chat_sessions SET title = %s WHERE id = %s", (title.strip()[:200] or "New chat", session_id))


def list_sessions(limit=50):
    """Chats that have at least one message, newest first."""
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT s.id, s.title, s.created_at, s.updated_at, count(c.id) AS messages
               FROM chat_sessions s JOIN conversations c ON c.session_id = s.id
               GROUP BY s.id ORDER BY s.updated_at DESC LIMIT %s""",
            (limit,),
        )
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_messages(session_id):
    """All questions/answers of a chat in order, with their stored checks."""
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT c.id, c.question, c.answer, c.budget_year, c.prompt_tokens, c.completion_tokens,
                      c.response_time, c.cost, c.timestamp,
                      (SELECT relevance FROM feedback WHERE conversation_id = c.id AND source = 'judge' ORDER BY id DESC LIMIT 1),
                      (SELECT explanation FROM feedback WHERE conversation_id = c.id AND source = 'judge' ORDER BY id DESC LIMIT 1),
                      (SELECT relevance FROM feedback WHERE conversation_id = c.id AND source = 'number_check' ORDER BY id DESC LIMIT 1),
                      (SELECT explanation FROM feedback WHERE conversation_id = c.id AND source = 'number_check' ORDER BY id DESC LIMIT 1),
                      (SELECT score FROM feedback WHERE conversation_id = c.id AND source = 'user' ORDER BY id DESC LIMIT 1)
               FROM conversations c WHERE c.session_id = %s ORDER BY c.timestamp, c.id""",
            (session_id,),
        )
        out = []
        for r in cur.fetchall():
            out.append({
                "conversation_id": r[0], "question": r[1], "answer": r[2], "budget_year": r[3],
                "prompt_tokens": r[4], "completion_tokens": r[5], "response_time": r[6], "cost": r[7],
                "timestamp": r[8], "relevance": r[9], "explanation": r[10],
                "num_status": r[11], "num_missing": [n for n in (r[12] or "").split(", ") if n],
                "user_score": r[13],
            })
        return out


def delete_session(session_id):
    """Delete a chat with all its messages and their feedback."""
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM feedback WHERE conversation_id IN (SELECT id FROM conversations WHERE session_id = %s)",
            (session_id,),
        )
        cur.execute("DELETE FROM conversations WHERE session_id = %s", (session_id,))
        cur.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))
