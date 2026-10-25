# SQL + RAG agent over Chinook

An agent built with **LangGraph** that answers natural-language questions about a music store (the [Chinook](https://github.com/lerocha/chinook-database) database). It explores the schema, writes and runs SQL queries and recovers from its own errors, and it combines the sales data with a **RAG** layer over the Wikipedia articles of the best-selling artists, citing its sources. Everything runs **locally** with [Ollama](https://ollama.com) models (`qwen3:14b` for the agent, `qwen3-embedding:0.6b` for the embeddings).

| Evaluation | Questions | Result |
|---|---|---|
| SQL agent (execution accuracy) | 26, 3 runs | **93.6 %** (92.3 – 96.2 %) |
| Full agent: SQL + Wikipedia, automatic checks | 17, 3 runs | **82.4 %** (from 53 % in the first version) |

Example run (the agent answers in the language of the question):

```
User: ¿Cuál es el artista que más dinero ha generado en ventas y dónde se formó?
      (Which artist has generated the most revenue, and where was the band formed?)

  → describe_tables(["InvoiceLine", "Track", "Album", "Artist"])
  → execute_query("SELECT Artist.Name, SUM(InvoiceLine.UnitPrice * InvoiceLine.Quantity) ... LIMIT 1")
  → search_wikipedia(query="Where was Iron Maiden formed?", artist_name="Iron Maiden")

Agent: El artista que más dinero ha generado en ventas es Iron Maiden, con un total de $138.6 en
       ingresos. Iron Maiden fue formado en Leyton, East London, Inglaterra, en el otoño de 1975
       por el bajista Steve Harris [1].

       Sources:
       [1] https://en.wikipedia.org/wiki/Iron_Maiden
```

## Architecture

A ReAct agent implemented as a LangGraph graph: at each step the model decides whether to call tools or answer.

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
| `execute_query` | Runs a SQL query on a read-only connection and returns at most 20 rows |
| `search_wikipedia` | Semantic search over the artists' Wikipedia articles, optionally filtered by artist; returns the matching chunks with their id and source URL |

## Data pipeline (RAG)

1. **Ingestion** (`descargar_wikipedia.py`): downloads from the Wikipedia API the articles of the 20 artists with the most revenue in the database, respecting the API rate limits. Chinook also sells TV series, so the database itself is used to disambiguate: if all of an "artist's" tracks are TV genres, it looks for the series (*Lost* is the TV series, not the band *Lost*).
2. **Chunking** (`chunking_script.py`): structure-aware, by sections and subsections. Short sections are merged and long ones split by paragraph (300–2,000 characters). Every chunk starts with a context header (artist and section) and carries metadata: `artist_id` (which links it to the SQL database), section and source URL. 810 chunks.
3. **Indexing** (`rag.py`): embeddings with `qwen3-embedding:0.6b` into a persistent Chroma collection. Chunk ids are stable (`{artist_id}_{n}`) and written with `upsert`, so re-indexing is idempotent. The collection name includes the embedding model, so vectors from different models are never mixed.
4. **Search** (`rag.py`): optional filter by artist, case-insensitive. If the artist has no articles (e.g. a band member like Freddie Mercury), it searches all artists and warns that the results may not be about that artist instead of returning nothing, so the model gets the answer from the documents and not from its own knowledge.

A quick check of cross-language retrieval: the same 6 questions in English and Spanish return the same top-3 chunks in 5 of 6 cases, with slightly higher distances in Spanish. The agent is asked to search in English.

## Design decisions

- **Safety by design, not by prompt.** The SQL connection is opened read-only (`mode=ro`): even if the model generates a `DELETE` or a `DROP`, SQLite rejects it. Write attempts are logged as warnings.
- **Errors the model can act on.** SQL errors (unknown column, syntax…) are returned to the model with the original message, so it can fix the query. Unexpected errors are not swallowed.
- **Context control.** `execute_query` returns at most 20 rows and says when it truncates, nudging the model towards `COUNT`, `GROUP BY` or `LIMIT`. Ollama silently truncates anything that doesn't fit in the context window (4096 tokens by default), so the context size is set explicitly (`num_ctx`).
- **Schema with relationships.** `describe_tables` returns the `CREATE TABLE` statement instead of a plain column list, so the model sees the foreign keys, plus sample rows so it sees the real data formats. It accepts several tables per call, and the table names are in the system prompt, to save loop iterations.
- **SQL injection in the tools.** Table names are validated against `sqlite_master` with parameterized queries before being used.
- **Concurrency.** `ToolNode` runs tool calls in parallel threads: each SQL call opens its own connection (SQLite connections can't be shared between threads), while the Chroma client is created once before the agent starts (creating several at the same time fails).
- **Citations: the model chooses, the code formats.** The model cites the ids of the chunks it used (`[90_2]`); the agent replaces them with numbers and appends the URLs, dropping any id the model never received. URLs can't be invented or mistyped.
- **Workarounds for `qwen3` on Ollama**, in the model node: the final answer sometimes arrives in the reasoning field with empty content, and tool calls are sometimes written as text (`<tool_call>{...}</tool_call>`) instead of real calls, which would end the agent. Both are fixed in code, in that order, because the second one can hide inside the first.

## Evaluation

### SQL agent

`evaluar_sql.py` runs the agent on **26 questions**, each with a hand-written reference query: counts, rankings, date aggregations, `LEFT JOIN` / `NOT EXISTS`, self-joins (`Employee.ReportsTo`) and relationships with non-obvious names (`Customer.SupportRepId`). Questions with ties or ambiguous wording were discarded.

**Main metric: execution accuracy**, as in the Spider and BIRD benchmarks. The last query the agent ran is re-executed and its result is compared with the reference result (extra columns are allowed, numbers are compared with a 0.01 tolerance, and row order only counts when the question asks for it).

`qwen3:14b` (Ollama, `temperature=0`, `num_ctx=16000`), 3 runs × 26 questions:

| Metric | Mean | Range across runs |
|---|---|---|
| **Execution accuracy** | **93.6 %** | 92.3 – 96.2 % |
| Correct final answer (natural language) | 97.5 % | 96.2 – 100 % |
| Tool calls per question | 1.9 | |
| Time per question (RTX 4060 Ti 16 GB) | 36 s | |

Only 2 of the 26 questions failed in any run: one where the agent took the (correct) answer from the sample rows of `describe_tables` instead of running a query, so there was no SQL to check, and one where it wrote the tool call as text and stopped (fixed later, see the design decisions). Detail: [`results/eval_sql.json`](results/eval_sql.json). This evaluation was run before the RAG layer was added, with the SQL-only version of the agent.

### Full agent (SQL + RAG)

`evaluar_agente_global.py` runs the complete agent on **17 questions** of five types: Wikipedia only, SQL then Wikipedia (*"which artist sells the most, and where was the band formed?"*), Wikipedia then SQL (*"which band did Freddie Mercury sing in, and how many of its songs does the store have?"*), artists in the store without Wikipedia articles, and SQL only. Every reference answer was checked against the database and the downloaded articles, not against general knowledge.

Each answer goes through automatic checks, and counts as correct only if it passes every check that applies to it:

| Check | What it verifies |
|---|---|
| Key facts | The answer contains the expected values (text without accents or case, numbers with a 0.01 tolerance) |
| No forbidden facts | It doesn't state facts the model knows but the documents don't contain (e.g. *Boston* for Aerosmith, which has no articles) |
| Tools | It called the tools the question needs, and not the ones it doesn't (e.g. no Wikipedia for a sales ranking) |
| Cites the source | It cites the Wikipedia article when the answer comes from the articles |
| Grounded numbers | Every number in the answer appears in a tool result or in the question, to catch invented results |

**Iterations.** Each change was measured with one run (`qwen3:14b`, `temperature=0`); the per-question detail of every version is in `results/`.

| Version | Correct | Wikipedia only | SQL → Wikipedia | Wikipedia → SQL | No articles | SQL only |
|---|---|---|---|---|---|---|
| [Baseline](results/eval_agente_baseline.json): *"only use Wikipedia when explicitly asked"* | **9/17 (53 %)** | 5/6 | 0/3 | 0/3 | 1/2 | 3/3 |
| [v2](results/eval_agente_v2.json): says what the database does *not* contain and when Wikipedia is *required* | **13/17 (76 %)** | 6/6 | 2/3 | 0/3 | 2/2 | 3/3 |
| [v3](results/eval_agente_v3.json): says the store also sells TV series; ambiguous question reworded | **14/17 (82 %)** | 5/6 | 3/3 | 1/3 | 2/2 | 3/3 |
| [v4](results/eval_agente_v4.json): numbered citations (the model cites chunk ids, the code adds the URLs) | **13/17 (76 %)** | 6/6 | 1/3 | 1/3 | 2/2 | 3/3 |
| [v5](results/eval_agente_v5.json): tool calls written as text are recovered | **14/17 (82 %)** | 6/6 | 2/3 | 1/3 | 2/2 | 3/3 |

What the iterations showed:
- **Prompt rules can overshoot.** The baseline rule kept Wikipedia out of the sales questions (3/3), but the agent then avoided it in mixed questions too: asked where U2 was formed, it looked in the database and answered that it *"can't be determined"*. Telling it what the database does **not** contain fixed most of it (v2), and that rule in turn made it assume the store sells no TV series, fixed in v3.
- **The checks catch what reading a few answers doesn't.** Invented results (*"the query returned 32 Queen songs"* without having run any query), facts from the model's own knowledge (*"AC/DC was formed in Sydney"*), a placeholder used as an argument (`artist_name="[Artist Name]"`) and an ambiguous question in the evaluation itself.
- **Not every fix is a prompt.** Citations (v4) and the recovery of tool calls written as text (v5) are code. The first version of the v5 fix didn't work: the call was inside the reasoning field, where the older workaround took it as the final answer.
- **The evaluation needed fixing too:** the number extraction read *"October 28, 1981"* as a single number and marked a correct answer as ungrounded.

**Current result (v5, 3 runs): 82.4 %**, 14/17 in each run, with every answer that uses Wikipedia citing its source. Detail: [`results/eval_agente.json`](results/eval_agente.json). The 3 failures are all on the SQL side of mixed questions: an invented count (*"11 Metallica albums"* without running a query; the real value is 10), a revenue query that summed whole invoices instead of the invoice lines, and a question about the best-selling *album* answered with the best-selling *artist*.

## Limitations and future work

**To read the numbers correctly:**
- Chinook is a well-known public database and the model has probably seen it during training (contamination), so results on a private database would likely be lower.
- 26 and 17 questions are small samples, and the iterations were compared with one run each, so a difference of one question between versions is not conclusive. The 3 runs of v5 gave identical answers, but the SQL evaluation, also at `temperature=0`, did vary between runs.
- The full-agent checks are automatic: they verify facts, tools, sources and numbers, but not the quality of open answers.

**Next steps, if the project continues:**
- **A verifier node before the end of the graph**, reusing the grounded-numbers check: if the final answer contains a number that no tool returned, send the model back to run the query (with a cap on retries). It targets the most serious remaining failure, invented results.
- **An LLM judge** for open answers (a different local model, e.g. `gpt-oss:20b` judging `qwen3:14b`), compared with a reference answer and validated against manual labels.
- **Retrieval evaluation** (hit rate @ k) to compare the distance metric (Chroma's default is L2; cosine is usual for embedding models), a larger embedding model (`qwen3-embedding:4b`), chunk overlap and semantic chunking.
- **SQL side:** include the schema of the involved tables in SQL error messages (the model sometimes guesses column names such as `Track.ArtistId`), limit the size of `describe_tables` output, and tell the model that sample rows are not the full table.
- **More questions**, and a version of the database with renamed tables and columns to measure the effect of contamination.

## How to run it

Requirements: [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com) and a GPU with about 12 GB of VRAM for `qwen3:14b`.

```bash
uv sync
ollama pull qwen3:14b
ollama pull qwen3-embedding:0.6b

# Data (not versioned)
uv run src/descargar_datos.py          # Chinook database → data/chinook.sqlite
uv run src/descargar_wikipedia.py 20   # articles of the 20 best-selling artists → data/wiki/
uv run src/rag.py                      # chunks and indexes them into Chroma → data/chroma/, then runs test searches

# Agent
uv run src/agente.py "¿Quién era el batería de Led Zeppelin?"   # your own question
uv run src/agente.py                                            # the example questions

# Evaluation
uv run src/evaluar_sql.py qwen3:14b 3            # SQL agent, 3 runs → results/eval_sql.json
uv run src/evaluar_agente_global.py qwen3:14b 3  # full agent, 3 runs → results/eval_agente.json
```

Code comments are in English; variable names and the evaluation questions are in Spanish.

## Project structure

```
src/
├── agente.py                 # tools, LangGraph agent and model-node workarounds
├── rag.py                    # indexing (embeddings + Chroma) and search
├── chunking_script.py        # structure-aware chunking
├── descargar_datos.py        # downloads the Chinook database
├── descargar_wikipedia.py    # downloads and disambiguates the artists' articles
├── evaluar_sql.py            # SQL agent evaluation (execution accuracy)
└── evaluar_agente_global.py  # full agent evaluation (automatic checks)
results/                      # evaluation results, one file per version
data/                         # downloaded data (not versioned)
```

## Data and licenses

- Code: MIT license.
- [Chinook Database](https://github.com/lerocha/chinook-database), by Luis Rocha: MIT license.
- [Wikipedia](https://en.wikipedia.org) articles: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Every chunk keeps the URL of its source article, and the agent cites it.

The data is not included in the repository; it is downloaded by the scripts.
