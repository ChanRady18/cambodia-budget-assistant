# --- data -------------------------------------------------------------
data:            ## rebuild documents/all_documents.json from data/*.csv
	uv run python build_documents.py

eval:            ## check that search finds the right document
	uv run python eval_retrieval.py -v

# --- local run (Postgres in Docker, app on your machine) --------------
postgres:        ## start only Postgres in the background
	docker compose up -d postgres

init-db:         ## create tables (safe to re-run; use reset-db to wipe)
	uv run python db_init.py

reset-db:
	uv run python db_init.py --reset

ask:             ## one question from the terminal: make ask Q="..."
	uv run python assistant.py "$(Q)"

chat:            ## web UI at http://localhost:8501
	uv run streamlit run app.py

fake-traffic:    ## fill the dashboard with fake conversations (Ctrl+C to stop)
	uv run python generate_data.py

# --- everything in Docker ---------------------------------------------
up:              ## Postgres + app + Grafana
	docker compose up -d --build

down:
	docker compose down

.PHONY: data eval postgres init-db reset-db ask chat fake-traffic up down
