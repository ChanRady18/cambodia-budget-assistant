FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .

# create tables if missing, then start the app
CMD ["sh", "-c", "python db_init.py && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
