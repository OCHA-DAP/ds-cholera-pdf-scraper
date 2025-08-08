---
applyTo: "**"
---

# Repository Context & Goal
This repository contains a baseline DataFrame/table extracted from PDFs. The current objective is to replicate or improve this extraction using OpenAI LLMs. Additionally, you need to:
- Download all **historical PDFs** locally and upload them to blob storage using `import ocha_stratus as stratus`.
- Ingest these PDFs via the LLM-based pipeline.
- Compare LLM outputs against baseline extraction.
- If successful, support a **production pipeline** that downloads new PDFs weekly, processes them via LLM, and merges with historical data.

---

## Project Structure Expectations
- `scripts/download_historical_pdfs.py`: downloads all existing PDFs and uploads to blob via `stratus`.
- `src/llm_extract.py`: calls OpenAI API with PDF text or structured input to fetch table data.
- `src/parse_output.py`: parses LLM responses into a `pandas.DataFrame` matching baseline schema.
- `src/compare.py`: compares new LLM DataFrame vs baseline, reports discrepancies.
- `scripts/weekly_ingest.py`: orchestrates weekly download, LLM processing, and update of the database.
- `tests/`: pytest tests validating each module, including edge cases and integration flows.

---

## Dependencies & Tooling
- Python 3.11.4 (via pyenv).
- Required libraries: `openai`, `pandas`, `pytest`, and `ocha_stratus`.
- Use Black (88‑char), flake8, mypy.
- NEVER ever fix linting with LLM AI, always leave to other tools

---

## `ocha_stratus` Usage Guidance
- Install via pip (`pip install ocha-stratus`) :contentReference[oaicite:1]{index=1}.
- Blob operations:
  ```python
  import ocha_stratus as stratus
  df = stratus.load_csv_from_blob("file.csv", stage="dev")
  stratus.upload_csv_to_blob(df, "file.csv", stage="dev")
