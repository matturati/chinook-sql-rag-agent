# SQL + RAG agent over Chinook

An agent built with **LangGraph** that answers natural-language questions about a music store (the [Chinook](https://github.com/lerocha/chinook-database) database). The agent explores the schema, writes and runs SQL queries, recovers from its own errors and explains the result. It runs **fully locally** with [Ollama](https://ollama.com) models (`qwen3:14b` by default).

In progress: a **RAG** layer over the artists' Wikipedia articles, to answer questions that combine sales data with information about the artists.

Example run:

```
User: Which support employee has generated the most sales through their assigned customers?

  → describe_tables(["Employee", "Customer", "Invoice"])
  → execute_query("SELECT e.FirstName, e.LastName, SUM(i.Total) ... GROUP BY e.EmployeeId ...")

Agent: Jane Peacock, with $833.04 in sales through her assigned customers...
```

## Architecture

A ReAct agent implemented as a LangGraph graph: at each step the model decides whether to call a tool or answer.

```mermaid
graph TD;
    __start__([start]) --> model;
    model -.->|tool calls| tools;
    model -.->|answer| __end__([end]);
    tools --> model;
```

| Tool | What it does |
|---|---|
| `list_tables` | Lists the tables in the database |
| `describe_tables` | For a list of tables, returns their `CREATE TABLE` statement (columns, types, primary and foreign keys) and 5 sample rows |
| `execute_query` | Runs a SQL query and returns at most 20 rows |

## Design decisions

- **Safety by design, not by prompt.** The connection is opened read-only (`mode=ro`): even if the model generates a `DELETE` or a `DROP`, SQLite rejects it. Write attempts are logged as warnings.
- **Errors the model can act on.** SQL errors (unknown column, syntax…) are returned to the model as a `ToolException` with the original message, so it can fix the query. Unexpected errors are not swallowed.
- **Context control.** `execute_query` returns at most 20 rows and says when it truncates, nudging the model towards `COUNT`, `GROUP BY` or `LIMIT`. Ollama silently truncates anything that doesn't fit in the context window (4096 tokens by default), so the context size is set explicitly (`num_ctx`).
- **Fewer loop iterations.** `describe_tables` accepts several tables in a single call, and the table names are included in the system prompt.
- **Schema with relationships.** The tool returns the `CREATE TABLE` statement instead of a plain column list, so the model sees the foreign keys, plus sample rows so it sees the real data formats (dates, country names…).
- **SQL injection in the tools.** Table names are validated against `sqlite_master` with parameterized queries before being used.
- **Concurrency.** `ToolNode` runs tools in threads, so each call opens its own connection.

## Evaluation

`evaluar_sql.py` runs the agent on **26 questions**, each with a hand-written reference query: counts, rankings, date aggregations, `LEFT JOIN` / `NOT EXISTS`, self-joins (`Employee.ReportsTo`) and relationships with non-obvious names (`Customer.SupportRepId`). Questions with ties or ambiguous wording were discarded.

**Main metric: execution accuracy**, as in the Spider and BIRD benchmarks. The last query the agent ran is re-executed and its result is compared with the reference result (extra columns are allowed, numbers are compared with a 0.01 tolerance, and row order only counts when the question asks for it).

<!-- RESULTS -->

**Limitations**, to read the numbers correctly:
- Chinook is a well-known public database and the model has probably seen it during training (contamination), so results on a private database would likely be lower.
- 26 questions is a small sample.
- The metric only looks at the last query executed: if the agent answers correctly without running SQL (for example, from the sample rows returned by `describe_tables`), it counts as a failure.

## RAG layer (in progress)

- [x] `descargar_wikipedia.py`: downloads the articles of the best-selling artists. It disambiguates using the database itself: if all of an "artist's" tracks are TV genres, it looks for the TV series (*Lost* is the TV series, not the band *Lost*). It respects the Wikipedia API rate limits.
- [x] `chunking_script.py`: structure-aware chunking (sections and subsections), merging of short sections, splitting of long ones by paragraph (300–2,000 characters), a context header (artist and section) on every chunk, and metadata (`artist_id`, which links to the SQL database; section; source URL).
- [ ] Embeddings (`qwen3-embedding`) and vector store (Chroma).
- [ ] Search tool for the agent, and questions that combine SQL and RAG.
- [ ] Compare structure-aware chunking with *semantic chunking*.

## How to run it

Requirements: [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com) and a GPU with about 12 GB of VRAM for `qwen3:14b`.

```bash
uv sync
ollama pull qwen3:14b

uv run src/descargar_datos.py          # Chinook database → data/chinook.sqlite
uv run src/agente.py                   # runs the agent on an example question
uv run src/evaluar_sql.py qwen3:14b 3  # evaluation (3 runs) → results/eval_sql.json

# RAG layer
uv run src/descargar_wikipedia.py 20   # articles of the 20 best-selling artists → data/wiki/
uv run src/chunking_script.py
```

The code comments and the evaluation questions are in Spanish.

## Project structure

```
src/
├── agente.py               # tools and LangGraph agent
├── evaluar_sql.py          # evaluation (execution accuracy)
├── descargar_datos.py      # downloads the Chinook database
├── descargar_wikipedia.py  # downloads and disambiguates the artists' articles
└── chunking_script.py      # structure-aware chunking for the RAG layer
results/                    # evaluation results
data/                       # downloaded data (not versioned)
```

## Data and licenses

- Code: MIT license.
- [Chinook Database](https://github.com/lerocha/chinook-database), by Luis Rocha: MIT license.
- [Wikipedia](https://en.wikipedia.org) articles: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Every chunk keeps the URL of its source article.

The data is not included in the repository; it is downloaded by the scripts.
