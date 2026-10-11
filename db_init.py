import os
import psycopg
from datetime import datetime

DB_TIMEZONE = datetime.now().astimezone().tzinfo
print(f"Using timezone: {DB_TIMEZONE}")


def get_db_connection():
    # Hosted Postgres (Neon, Supabase, Render…) gives one connection string
    if os.getenv("DATABASE_URL"):
        return psycopg.connect(os.getenv("DATABASE_URL"))
    return psycopg.connect(
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        dbname=os.getenv("POSTGRES_DB", "budget_assistant"),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
    )

def init_db(drop=False):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if drop:
                cur.execute("DROP TABLE IF EXISTS conversations")

            # one row per chat; its messages are the conversations rows with this session_id
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id UUID PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT 'New chat',
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    updated_at TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    budget_year TEXT NOT NULL,
                    model TEXT NOT NULL,
                    instructions TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    response_time FLOAT NOT NULL,
                    cost FLOAT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
            # added later: safe on databases created by an older version (existing rows get NULL)
            cur.execute("""
                ALTER TABLE conversations
                ADD COLUMN IF NOT EXISTS session_id UUID REFERENCES chat_sessions(id)
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS conversations_session_idx ON conversations (session_id, timestamp)")
        conn.commit()
    finally:
        conn.close()


def init_feedback():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER REFERENCES conversations(id),
                    source TEXT NOT NULL,
                    relevance TEXT,
                    explanation TEXT,
                    score INTEGER,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
        conn.commit()
    finally:
        conn.close()

if __name__ == "__main__":
    import sys
    if "--reset" in sys.argv:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS feedback")
            cur.execute("DROP TABLE IF EXISTS conversations")
            cur.execute("DROP TABLE IF EXISTS chat_sessions")
        conn.commit()
        conn.close()
    init_db()
    init_feedback()
    print("Database initialized")