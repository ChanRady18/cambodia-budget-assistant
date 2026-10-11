import os

from openai import RateLimitError, APIStatusError, APIConnectionError
from dataclasses import asdict
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from assistant import create_assistant
from db_save import save_conversation
from db_feedback import save_feedback
from judge import evaluate_relevance, check_numbers
from rag_helper import detect_year
from db_query import get_conversations, get_stats
from db_init import DB_TIMEZONE
from db_sessions import (create_session, session_exists, touch_session, rename_session,
                         list_sessions, get_messages, delete_session)
from datetime import datetime, timedelta

load_dotenv()

# Streamlit Community Cloud keeps keys in "Secrets", not .env: copy them into env vars.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float)) and not os.getenv(_k):
            os.environ[_k] = str(_v)
except Exception:
    pass  # no secrets file (local / Docker): .env is used instead


def _flag(name):
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


SHOW_SETTINGS = not _flag("HIDE_SETTINGS")   # set HIDE_SETTINGS=true when hosting publicly
# Sidebar list of ALL saved chats. Everyone using the app sees it, so it is off by default
# when hosting publicly (HIDE_SETTINGS=true). Force it with SHOW_HISTORY=true/false.
SHOW_HISTORY = _flag("SHOW_HISTORY") if os.getenv("SHOW_HISTORY") else SHOW_SETTINGS

st.set_page_config(page_title="Budget Assistant", page_icon="📊", layout="wide")

# Optional password so strangers can't spend your API credits: set APP_PASSWORD
if os.getenv("APP_PASSWORD") and not st.session_state.get("authed"):
    st.title("Budget Assistant")
    pw = st.text_input("Password", type="password")
    if pw and pw == os.getenv("APP_PASSWORD"):
        st.session_state.authed = True
        st.rerun()
    elif pw:
        st.error("Wrong password")
    st.stop()


@st.cache_resource
def _create_tables():
    from db_init import init_db, init_feedback
    init_db()
    init_feedback()


_create_tables()   # safe to repeat: tables are only created if missing

from ingest import load_budget_data
YEARS = ["Auto"] + sorted({d["budget_year"] for d in load_budget_data()}, reverse=True)
DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-max-2026-06-08"

DEFAULT_INPUT_PRICE = 0.96
DEFAULT_OUTPUT_PRICE = 3.84

MODEL_PRESETS = {
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60, "base_url": "https://api.groq.com/openai/v1"},
    "qwen/qwen3.8-27b": {"input": 0.80, "output": 4.00, "base_url": "https://api.groq.com/openai/v1"},
    "qwen3.7-max-2026-06-08": {"input": 0.96, "output": 3.84},
    "qwen-plus": {"input": 0.40, "output": 1.20},
    "qwen-turbo": {"input": 0.05, "output": 0.20},
    "qwen-max": {"input": 1.60, "output": 6.40},
}

# Initialize session state for API credentials, model, and token pricing
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")
if "model_name" not in st.session_state:
    st.session_state.model_name = os.getenv("LLM_MODEL", DEFAULT_MODEL)
if "base_url" not in st.session_state:
    st.session_state.base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
if "input_token_price" not in st.session_state:
    st.session_state.input_token_price = float(os.getenv("INPUT_TOKEN_PRICE") or DEFAULT_INPUT_PRICE)
if "output_token_price" not in st.session_state:
    st.session_state.output_token_price = float(os.getenv("OUTPUT_TOKEN_PRICE") or DEFAULT_OUTPUT_PRICE)


@st.cache_resource
def get_assistant(api_key, model, base_url, input_price, output_price):
    return create_assistant(
        api_key=api_key,
        model=model,
        base_url=base_url,
        input_price_per_million=input_price,
        output_price_per_million=output_price
    )


# ---------------------------------------------------------------- chat sessions

def _section(ts):
    """Sidebar group for a chat's last activity."""
    today = datetime.now(DB_TIMEZONE).date()
    d = ts.astimezone(DB_TIMEZONE).date()
    if d == today:
        return "Today"
    if d == today - timedelta(days=1):
        return "Yesterday"
    if d > today - timedelta(days=7):
        return "Previous 7 days"
    if d > today - timedelta(days=30):
        return "Previous 30 days"
    return "Older"


def _open_session(session_id):
    """Load a chat (or start an empty one when session_id is None)."""
    st.session_state.session_id = session_id
    st.session_state.messages = get_messages(session_id) if session_id else []
    if session_id:
        st.query_params["session"] = session_id
    elif "session" in st.query_params:
        del st.query_params["session"]


def _sync_session_from_url():
    """?session=<uuid> in the URL reopens that chat (bookmark / refresh / shared link)."""
    wanted = st.query_params.get("session")
    if "session_id" not in st.session_state:
        _open_session(wanted if wanted and session_exists(wanted) else None)
    elif wanted and wanted != st.session_state.session_id and session_exists(wanted):
        _open_session(wanted)


def _sidebar():
    with st.sidebar:
        if st.button("➕ New chat", use_container_width=True, type="primary"):
            _open_session(None)
            st.rerun()

        sid = st.session_state.session_id
        if sid:
            st.caption(f"Session `{sid}`")
            with st.popover("✏️ Rename / 🗑 Delete", use_container_width=True):
                current = next((s["title"] for s in list_sessions(200) if str(s["id"]) == sid), "")
                new_title = st.text_input("Chat name", value=current, key=f"rename_{sid}")
                if st.button("Save name", key=f"save_name_{sid}"):
                    rename_session(sid, new_title)
                    st.rerun()
                st.divider()
                if st.button("🗑 Delete this chat", key=f"delete_{sid}"):
                    delete_session(sid)
                    _open_session(None)
                    st.rerun()

        if not SHOW_HISTORY:
            return
        sessions = list_sessions(limit=100)
        if not sessions:
            st.caption("No saved chats yet.")
            return
        group = None
        for s in sessions:
            g = _section(s["updated_at"])
            if g != group:
                st.markdown(f"**{g}**")
                group = g
            is_current = str(s["id"]) == sid
            label = ("▶ " if is_current else "") + s["title"]
            if st.button(label, key=f"open_{s['id']}", use_container_width=True,
                         help=f"{s['messages']} question(s) · {s['id']}"):
                _open_session(str(s["id"]))
                st.rerun()


def _render_checks(msg):
    rel_badge = {
        "CORRECT": "🟢 **CORRECT**",
        "PARTLY_CORRECT": "🟡 **PARTLY CORRECT**",
        "INCORRECT": "🔴 **INCORRECT**",
        "NOT_CHECKED": "⚪ **NOT CHECKED**",
    }.get(msg.get("relevance"), f"**{msg.get('relevance') or 'not checked'}**")
    with st.expander(f"⚖️ Judge: {rel_badge}", expanded=False):
        if msg.get("explanation"):
            st.write(f"**Explanation:** {msg['explanation']}")
        status = msg.get("num_status")
        if status == "OK":
            st.write("🔢 Number check: ✅ every number matches the budget documents")
        elif status == "MISMATCH":
            st.write("🔢 Number check: ⚠️ not found in the documents: "
                     + ", ".join(f"`{n}`" for n in msg.get("num_missing", [])))
        elif status:
            st.write("🔢 Number check: no numbers in the answer")
        st.caption(f"⏱️ {msg['response_time']:.2f}s · 📥 {msg['prompt_tokens']} · "
                   f"📤 {msg['completion_tokens']} tokens · 💰 ${msg['cost']:.5f}")

    cid = msg["conversation_id"]
    if msg.get("user_score") == 1:
        st.caption("You rated this 👍")
    elif msg.get("user_score") == -1:
        st.caption("You rated this 👎")
    else:
        c1, c2, _ = st.columns([1, 1, 6])
        if c1.button("👍", key=f"up_{cid}"):
            save_feedback(cid, "user", score=1)
            msg["user_score"] = 1
            st.rerun()
        if c2.button("👎", key=f"down_{cid}"):
            save_feedback(cid, "user", score=-1)
            msg["user_score"] = -1
            st.rerun()


def _llm_error_text(e):
    """Short, readable reason for an LLM API error."""
    if isinstance(e, RateLimitError):
        msg = str(e)
        try:
            wait = e.response.headers.get("retry-after")
        except Exception:
            wait = None
        if "Request too large" in msg or "exceed the enforced limit" in msg:
            return ("the request is larger than this model's per-minute limit on your Groq plan. "
                    "Use openai/gpt-oss-120b, or lower ANSWER_MAX_TOKENS / JUDGE_MAX_TOKENS in .env.")
        return f"Groq rate limit reached. Wait {wait or 'about 60'} seconds and ask again."
    if isinstance(e, APIConnectionError):
        return "cannot reach the LLM server. Check your internet connection and OPENAI_BASE_URL."
    if isinstance(e, APIStatusError):
        if e.status_code == 401:
            return "the API key was rejected (401). Check OPENAI_API_KEY in .env."
        if e.status_code == 404:
            return "model not found (404). Check LLM_MODEL in .env."
        return f"the LLM server returned an error ({e.status_code})."
    return str(e)


def _answer(question, year):
    """Run RAG + checks for one question in the current chat and store everything."""
    history = st.session_state.messages
    assistant = get_assistant(
        st.session_state.api_key, st.session_state.model_name, st.session_state.base_url,
        st.session_state.input_token_price, st.session_state.output_token_price,
    )
    assistant.budget_year = None if year == "Auto" else year
    assistant.model = st.session_state.model_name
    assistant.input_price_per_million = st.session_state.input_token_price
    assistant.output_price_per_million = st.session_state.output_token_price

    answer = assistant.rag(question, history=history)   # errors here are shown by chat_page
    record = assistant.last_call
    if not answer.strip():
        answer = ("⚠️ The model returned an empty answer — it probably ran out of output tokens. "
                  "Raise ANSWER_MAX_TOKENS in .env, or use openai/gpt-oss-120b.")

    # the chat gets its UUID when its first question is asked (no empty chats in the DB)
    if not st.session_state.session_id:
        _open_session(create_session())
    sid = st.session_state.session_id

    prev_q = history[-1]["question"] if history else ""
    saved_year = year if year != "Auto" else (detect_year(question) or detect_year(prev_q) or "unknown")
    conversation_id = save_conversation(record, question, saved_year, session_id=sid)
    touch_session(sid, first_question=question if not history else None)

    num_status, num_missing = check_numbers(answer, record.prompt)
    save_feedback(conversation_id, "number_check", relevance=num_status,
                  explanation=", ".join(num_missing) or None)

    judge_question = f"Earlier question: {prev_q}\nFollow-up: {question}" if prev_q else question
    try:
        relevance, explanation = evaluate_relevance(
            judge_question, answer, context=record.prompt,
            model=st.session_state.model_name, api_key=st.session_state.api_key,
            base_url=st.session_state.base_url,
        )
    except (RateLimitError, APIStatusError, APIConnectionError) as e:
        # the answer is already saved and shown; only the judge is skipped
        relevance, explanation = "NOT_CHECKED", f"Judge skipped: {_llm_error_text(e)}"
    save_feedback(conversation_id, "judge", relevance=relevance, explanation=explanation)

    st.session_state.messages.append({
        "conversation_id": conversation_id, "question": question, "answer": answer,
        "budget_year": saved_year, "prompt_tokens": record.prompt_tokens,
        "completion_tokens": record.completion_tokens, "response_time": record.response_time,
        "cost": record.cost, "relevance": relevance, "explanation": explanation,
        "num_status": num_status, "num_missing": num_missing, "user_score": None,
    })


def chat_page():
    _sync_session_from_url()
    _sidebar()

    header_col1, header_col2, header_col3 = st.columns([3, 1, 1])
    with header_col1:
        st.title("Budget Assistant")
        st.caption(
            f"🤖 **Model:** `{st.session_state.model_name}` | 💰 **Pricing:** `${st.session_state.input_token_price:.2f}` in / `${st.session_state.output_token_price:.2f}` out (per 1M) | 🌐 **API Base:** `{st.session_state.base_url}`"
        )
    with header_col2:
        st.write("")
        if st.button("📊 View Dashboard", use_container_width=True):
            st.switch_page(page_dashboard)
    with header_col3:
        st.write("")
        if SHOW_SETTINGS and st.button("⚙️ Settings", use_container_width=True):
            st.switch_page(page_settings)

    if not st.session_state.api_key:
        st.warning("⚠️ No API Key configured. Set OPENAI_API_KEY in .env (or Secrets), or enter one in Settings.")
        if SHOW_SETTINGS and st.button("👉 Go to Settings"):
            st.switch_page(page_settings)
        return

    year = st.selectbox("Budget year (Auto = detect from the question):", YEARS, index=0)

    if not st.session_state.messages:
        st.info("Ask about a ministry's budget, actual spending, execution rate or share of GDP — "
                "in Khmer or English. Follow-up questions like “and in 2023?” use the earlier messages.")

    for msg in st.session_state.messages:
        with st.chat_message("user"):
            st.markdown(msg["question"])
        with st.chat_message("assistant"):
            st.markdown(msg["answer"])
            _render_checks(msg)

    question = st.chat_input("e.g. តើក្រសួងសុខាភិបាលមានថវិកាប៉ុន្មាន ឆ្នាំ២០២៤? / How much did MPWT spend in 2024?")
    if question and question.strip():
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    _answer(question.strip(), year)
                except (RateLimitError, APIStatusError, APIConnectionError) as e:
                    st.error("⚠️ No answer: " + _llm_error_text(e))
                    st.caption(f"Model: `{st.session_state.model_name}`. Your question was not saved; send it again when ready.")
                    st.stop()
        st.rerun()


def dashboard_page():
    header_col1, header_col2, header_col3 = st.columns([3, 1, 1])
    with header_col1:
        st.title("Budget Assistant Dashboard")
    with header_col2:
        st.write("")
        if st.button("💬 Back to Chat", use_container_width=True):
            st.switch_page(page_chat)
    with header_col3:
        st.write("")
        if SHOW_SETTINGS and st.button("⚙️ Settings", use_container_width=True):
            st.switch_page(page_settings)

    stats = get_stats()

    col1, col2, col3, col4 = st.columns(4)
    avg_time = stats.avg_response_time or 0.0
    total_cost = stats.total_cost or 0.0
    avg_tokens = stats.avg_tokens or 0

    col1.metric("Total conversations", stats.total or 0)
    col2.metric("Avg response time", f"{avg_time:.2f}s")
    col3.metric("Total cost", f"${total_cost:.4f}")
    col4.metric("Avg tokens", f"{avg_tokens:.0f}")


    records = get_conversations(limit=100)
    df = pd.DataFrame([asdict(r) for r in records])

    st.subheader("Cost over time")
    if not df.empty and "timestamp" in df.columns and "cost" in df.columns:
        st.line_chart(df, x="timestamp", y="cost")
    else:
        st.info("No cost data available yet.")

    st.subheader("Response time over time")
    if not df.empty and "timestamp" in df.columns and "response_time" in df.columns:
        st.line_chart(df, x="timestamp", y="response_time")
    else:
        st.info("No response time data available yet.")

    st.subheader("Recent conversations")
    recent_records = get_conversations(limit=20)

    for record in recent_records:
        st.write(f"**{record.prompt[:80]}...**")
        st.write(f"{record.answer[:200]}...")
        st.write(f"Time: {record.response_time:.2f}s | Cost: ${record.cost:.4f}")
        st.divider()


def settings_page():
    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.title("⚙️ LLM & API Settings")
    with header_col2:
        st.write("")
        if st.button("💬 Back to Chat", use_container_width=True):
            st.switch_page(page_chat)

    st.markdown(
        "Configure your LLM model and API credentials here. These settings are dynamically applied to the Budget Assistant and Relevance Judge without restarting the application."
    )

    with st.form("settings_form"):
        api_key_input = st.text_input(
            "API Key",
            value="",
            type="password",
            placeholder="A key is already set (hidden)" if st.session_state.api_key else "Paste your API key",
            help="Leave empty to keep the current key. The saved key is never shown here."
        )

        model_input = st.text_input(
            "Model Name",
            value=st.session_state.model_name,
            help="e.g. qwen3.7-max-2026-06-08, qwen-plus, qwen-turbo, qwen-max"
        )

        base_url_input = st.text_input(
            "Base URL",
            value=st.session_state.base_url,
            help="Default: https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )

        st.markdown("##### 🪙 Token Pricing ($ / 1 Million Tokens)")
        price_col1, price_col2 = st.columns(2)
        with price_col1:
            input_price_input = st.number_input(
                "Input Token Price ($)",
                min_value=0.0,
                value=float(st.session_state.input_token_price),
                step=0.01,
                format="%.4f",
                help="Price per 1M input prompt tokens"
            )
        with price_col2:
            output_price_input = st.number_input(
                "Output Token Price ($)",
                min_value=0.0,
                value=float(st.session_state.output_token_price),
                step=0.01,
                format="%.4f",
                help="Price per 1M completion output tokens"
            )

        submitted = st.form_submit_button("💾 Save Settings", use_container_width=True)
        if submitted:
            if api_key_input.strip():
                st.session_state.api_key = api_key_input.strip()
            st.session_state.model_name = model_input.strip()
            st.session_state.base_url = base_url_input.strip()
            st.session_state.input_token_price = float(input_price_input)
            st.session_state.output_token_price = float(output_price_input)
            st.cache_resource.clear()
            st.success("✅ Settings updated successfully!")

    st.subheader("Quick Model Presets")
    st.caption("Click to load default model name and pricing presets:")
    preset_cols = st.columns(len(MODEL_PRESETS))
    for i, (m_name, p_info) in enumerate(MODEL_PRESETS.items()):
        with preset_cols[i]:
            provider = "Groq" if "groq" in p_info.get("base_url", "") else "DashScope"
            btn_label = f"**{m_name.split('-202')[0]}** ({provider})\n\n`${p_info['input']:.2f}` / `${p_info['output']:.2f}`"
            if st.button(btn_label, use_container_width=True, key=f"preset_{i}"):
                st.session_state.model_name = m_name
                st.session_state.base_url = p_info.get("base_url", DEFAULT_BASE_URL)
                st.session_state.input_token_price = float(p_info["input"])
                st.session_state.output_token_price = float(p_info["output"])
                st.cache_resource.clear()
                st.rerun()


page_chat = st.Page(chat_page, title="Budget Assistant", icon="💬", default=True)
page_dashboard = st.Page(dashboard_page, title="Dashboard", icon="📊", url_path="dashboard")
page_settings = st.Page(settings_page, title="Settings", icon="⚙️", url_path="settings")

pg = st.navigation([page_chat, page_dashboard] + ([page_settings] if SHOW_SETTINGS else []))
pg.run()