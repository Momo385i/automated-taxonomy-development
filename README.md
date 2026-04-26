# mas_taxonomy

A multi-agent system (built on **LangGraph** + **LangChain**) that develops empirical
taxonomies from a corpus of scientific papers, following the
**Nickerson, Varshney & Muntermann (2013)** taxonomy-development method. The
pipeline ingests PDFs, extracts their text, and runs an iterative agent graph
(consultation → agent empirical worker → consolidator → validator → optional interaction step) that
incrementally builds, refines, and evaluates a taxonomy across multiple iterations.

---

## Table of contents

1. [Requirements](#1-requirements)
2. [Quick start (TL;DR)](#2-quick-start-tldr)
3. [Detailed setup](#3-detailed-setup)
4. [Configuring API keys](#4-configuring-api-keys)
5. [Running the pipeline](#5-running-the-pipeline)
6. [Iterative workflow: adding new papers per iteration](#6-iterative-workflow-adding-new-papers-per-iteration)
7. [Project layout](#7-project-layout)
8. [Included example runs](#8-included-example-runs)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Requirements

- **Python 3.12** (pinned via `.python-version`; `>=3.12,<3.13`)
- **[uv](https://docs.astral.sh/uv/)** as the package / environment manager
  (recommended; resolves `uv.lock` exactly)
- An API key for **at least one** of:
  OpenAI, Anthropic, Google Gemini, or Google Vertex AI

That's it. No databases, no external services.

---

## 2. Quick start (TL;DR)

```bash
# 1. Clone
git clone <your-fork-url> mas_taxonomy
cd mas_taxonomy

# 2. Install Python deps into a local .venv (uv reads pyproject.toml + uv.lock)
uv sync

# 3. Create your .env from the template and add at least one API key
cp .env.example .env        # PowerShell: Copy-Item .env.example .env
#   then edit .env and set e.g. OPENAI_API_KEY=sk-...

# 4. Drop the PDFs you want to analyse into data/input_pdfs/
#    (any number of *.pdf files; the folder is created automatically on first
#     run via mas_taxonomy/config.py if it does not yet exist)

# 5. Configure & start a run (interactive)
uv run python -m mas_taxonomy configure-run --run-id run_demo
```

The `configure-run` command will walk you through the intake questionnaire,
ingest the PDFs, and start the graph. Press 1 / 2 at the prompts to make
choices. When iteration 1 finishes, you decide whether to continue with
iteration 2, etc. (see [section 6](#6-iterative-workflow-adding-new-papers-per-iteration)).

---

## 3. Detailed setup

### 3.1 Install `uv`

`uv` is a fast Python package manager that handles virtualenvs, lockfiles, and
Python interpreters in one tool.

- **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **macOS / Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

Restart your shell, then verify:

```bash
uv --version
```

### 3.2 Install Python 3.12 (if you don't have it)

```bash
uv python install 3.12
```

`uv` will pick this up automatically because of `.python-version`.

### 3.3 Sync dependencies

From the project root:

```bash
uv sync
```

This creates `.venv/` next to `pyproject.toml`, installs all packages from
`uv.lock`, and installs the project itself in editable mode (so the
`mas_taxonomy` CLI is available via `uv run`).

You do **not** need to manually `activate` the venv — prefix any command with
`uv run` and uv will use the right environment. (You may still activate it the
classic way if you prefer: `.venv\Scripts\Activate.ps1` on Windows, or
`source .venv/bin/activate` elsewhere.)

---

## 4. Configuring API keys

The project loads its configuration from a `.env` file in the project root
(see `src/mas_taxonomy/config.py`).

1. Copy the template:
   ```bash
   cp .env.example .env       # PowerShell: Copy-Item .env.example .env
   ```
2. Open `.env` and fill in **at least one** of the following:
   - `OPENAI_API_KEY=sk-...`
   - `ANTHROPIC_API_KEY=sk-ant-...`
   - `GEMINI_API_KEY=...`
   - or Vertex AI (`VERTEX_PROJECT`, `VERTEX_LOCATION`, `VERTEX_CREDENTIALS_FILE`)

3. (Optional) reorder `LLM_PROVIDER_PRIORITY` to choose which provider is
   preferred when multiple keys are set. The first provider in the list with a
   non-empty key wins.

4. (Optional) **Change the model** for the chosen provider. Uncomment and set
   any of the `DEFAULT_MODEL_*` variables in your `.env`:

   ```dotenv
   # Override the OpenAI model (default: gpt-5.4-2026-03-05)
   DEFAULT_MODEL_OPENAI=gpt-5-mini
   # Override the Anthropic model (default: claude-haiku-4-5)
   DEFAULT_MODEL_ANTHROPIC=claude-sonnet-4-5
   # Override the Google Gemini model (default: gemini-2.5-pro)
   DEFAULT_MODEL_GEMINI=gemini-2.5-flash
   # Override the Vertex AI model (default: gemini-2.5-flash)
   DEFAULT_MODEL_VERTEXAI=gemini-2.5-pro
   ```

   Pick a model that is available on your account/region and that supports
   tool use / structured output — the agents rely on it. If the variable is
   unset, the default shown above is used. (See
   `src/mas_taxonomy/config.py` for the full list of fields.)

The `.env` file and any Vertex AI service-account JSON (matched by the
`gen-lang-*` pattern) are **git-ignored** and will never be pushed.

---

## 5. Running the pipeline

The CLI is exposed as a Typer app with four commands:

| Command | Purpose |
|---|---|
| `list-inputs`   | List the PDFs currently in `data/input_pdfs/` |
| `configure-run` | Interactive: define topic, meta-characteristic, ending conditions; chained to ingest + run-graph |
| `ingest`        | Stand-alone: extract text from PDFs into a run's `extracted/` and write `input_manifest.yaml` |
| `run-graph`     | Stand-alone: execute the LangGraph pipeline for an existing run directory |

Run any command with:

```bash
uv run python -m mas_taxonomy <command> [options]
```

### 5.1 The recommended path: `configure-run`

```bash
uv run python -m mas_taxonomy configure-run --run-id run_demo
```

What happens:

1. **Intake questionnaire** — you describe the target users, purpose,
   types of objects to classify, and any restrictions.
2. **Optional LLM consultation** — choice "2" lets an agent help you sharpen
   the topic and meta-characteristic; choice "1" skips it.
3. **Ending conditions** — Nickerson defaults are pre-loaded; you can add or
   refine objective and subjective ending conditions interactively.
4. **Run config saved** to `data/runs/run_demo/run_config.yaml`.
5. **Ingest** — every PDF in `data/input_pdfs/` is hashed, extracted with
   PyMuPDF, and written to `data/runs/run_demo/extracted/<paper>.txt`. The
   resulting manifest is `data/runs/run_demo/input_manifest.yaml`.
6. **Graph execution** — the LangGraph pipeline runs iteration 1, then asks
   whether to continue. See [section 6](#6-iterative-workflow-adding-new-papers-per-iteration).

### 5.2 The split path: `ingest` then `run-graph`

If you prefer to set things up step by step:

```bash
uv run python -m mas_taxonomy ingest    --run-id run_demo
uv run python -m mas_taxonomy run-graph --run-id run_demo
```

(`configure-run` must still have created the run directory and config first,
or the graph won't have a topic to work on.)

### 5.3 Listing inputs

```bash
uv run python -m mas_taxonomy list-inputs
```

---

## 6. Iterative workflow: adding new papers per iteration

The taxonomy is built incrementally. Between iterations you typically expand
the corpus with new papers so the graph can refine its dimensions and
characteristics on more evidence.

**The rule of thumb:**

> Before each new iteration, place the additional PDFs you want to include
> for that iteration into `data/input_pdfs/`. Existing PDFs already covered
> by the manifest are detected by their SHA-256 hash and **skipped**, so it
> is safe to leave previous papers in place.

Concretely:

1. **Iteration 1:** put your initial batch of PDFs into `data/input_pdfs/` and
   run `configure-run` (or `ingest` + `run-graph`).
2. When the iteration finishes, the CLI asks whether to continue:
   - choose **Continue** to start iteration 2.
3. **Before continuing**, drop any *new* PDFs for iteration 2 into
   `data/input_pdfs/`. The next ingest pass will pick them up, extract them,
   and tag them with `iteration_added: 2` in the manifest. Already-known
   papers are not re-extracted.
4. Repeat for iteration 3 (and further) until the ending conditions are met
   or you choose to finish.

Per-iteration outputs are written to `data/runs/<run_id>/outputs/iter_NNN/`
(empirical / consolidator / validator / interaction artifacts plus a
`graph_state_iter_NNN.yaml`). The final taxonomy of the last iteration is
shown in the CLI summary at the end.

---

## 7. Project layout

```
mas_taxonomy/
├── src/mas_taxonomy/        ← Python package (CLI, graph, IO, logging, config)
│   ├── cli.py               ← Typer commands (list-inputs / configure-run / ingest / run-graph)
│   ├── config.py            ← Settings (paths + env vars via pydantic-settings)
│   ├── graph/skeleton.py    ← LangGraph pipeline (workers, consolidator, validator, …)
│   ├── io/pdf_loader.py     ← PDF extraction (PyMuPDF / pymupdf4llm)
│   ├── llm_utils.py         ← Provider/model resolution + retry/cost helpers
│   ├── run_config.py        ← Run-config schema (topic, ending conditions, etc.)
│   └── …
├── data/
│   ├── input_pdfs/          ← PUT YOUR PDFs HERE (git-ignored)
│   └── runs/                ← One subdir per run (only example runs are shipped)
├── pyproject.toml           ← Project + dependency declarations
├── uv.lock                  ← Pinned dependency graph (used by `uv sync`)
├── .env.example             ← Template for your API keys (copy to .env)
├── .python-version          ← Python version pin (3.12)
└── README.md
```

---

## 8. Included example runs

For reproducibility, six finished runs are shipped under `data/runs/`:

- `run_22RIGORA`, `run_22RIGORB`, `run_22RIGORC`, `run_22RIGORD`, `run_22RIGORE`
  — five rigour-evaluation runs on the same corpus.
- `run_PGHD` — taxonomy of the integration of Patient-Generated Health Data
  (PGHD) into clinical systems.

Each run directory contains its `run_config.yaml`, `input_manifest.yaml`
(metadata of the papers used: file name, SHA-256, page count,
`iteration_added`), `logs/`, and per-iteration outputs (`outputs/iter_001/`,
`iter_002/`, `iter_003/`). The PDF files themselves **and the extracted
text** of those papers are **not** shipped — the source papers are
copyrighted material whose redistribution is not covered by this repo's
license. Both can be regenerated locally by placing your own PDFs in
`data/input_pdfs/` and running `ingest`.

---

## 9. Troubleshooting

- **`No valid API key found for any provider`** — your `.env` has no usable
  key for any provider listed in `LLM_PROVIDER_PRIORITY`. Set at least one.
- **`no pdfs found in: …/data/input_pdfs`** — the input folder is empty.
  Put `*.pdf` files there before running `ingest` / `configure-run`.
- **`uv: command not found`** — install `uv` (see [section 3.1](#31-install-uv))
  and restart your shell.
- **Re-running iteration with new papers does nothing** — check that the new
  PDFs are physically in `data/input_pdfs/` and have a different SHA-256
  hash than already-ingested ones (i.e. the file content actually differs).
- **Vertex AI auth errors** — make sure `VERTEX_CREDENTIALS_FILE` points to
  the JSON key file relative to the project root and that the file is
  readable. The file itself is git-ignored.

---
