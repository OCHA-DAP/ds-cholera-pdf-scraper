# Prompt Engineering Logging Module Instructions

##  Objectives

- Implement a system that captures and organizes prompts used in LLM workflows.
- Implement only when there is a successful call to an LLM.
- Enable reproducibility, auditing, iteration, and debugging of prompt-based interactions

##  Functional Requirements

1. **Prompt Versioning**
   - Store each prompt template (system, user, examples) in files or a database with metadata:
     - `version`: semantic identifier (e.g., `v1.0.0`)
     - `created_at`: ISO timestamp
     - `description`: brief context (e.g. “Kenya cholera extraction v2”)
   - Enable easy retrieval and rollback between prompt versions

2. **Logging Each Model Call**
   - Log the full prompt content used (including system + user components)
   - Record:
     - Model name and version
     - All parameters (temperature, max_tokens, stop tokens, etc.)
     - Timestamp of the call
   - Capture and store the raw model response, including cases where parsing fails

3. **Performance Metrics**
   - Define and compute relevant execution metrics per prompt version:
     - Number of records extracted
     - Parsing success/failure rate
     - Completeness compared to expected count
   - Attach metrics to the logged call entry

4. **Storage Format & Access**
   - Store logs in structured format (e.g. JSONL, SQLite, or proper logging DB)
   - Include:
     - Metadata (version, date, model, parameters)
     - Prompt text
     - Model response text
     - Parsed output summary (e.g. record count, error flag)
   - Ensure prompt files and logs are commit-controlled using Git or similar

##  Technical Guidance

- Use a **PromptManager** class or module that:
  - Loads prompt templates with version metadata
  - Provides the current version and its content
  - Handles fallback or upgraded versions in case of failure
- Use a **PromptLogger** utility that:
  - Saves each interaction (prompt + response + metadata + metrics)
  - Supports querying or filtering logs by prompt version, date, task type
- For large model outputs or errors, include truncated or salvaged response text

##  Suggested Workflow

1. Developer updates the prompt file with new content and bump version.
2. Run the prompt through the pipeline—PromptManager loads it.
3. Send to LLM → receive response.
4. PromptLogger logs:
   - Version, timestamp, parameters, model used
   - Full prompt and model output
   - Parsing metric(s)
5. Review logs to compare performance across versions and adjust accordingly.

---

By following this structure, your prompt engineering becomes reproducible, traceable, and continuously optimizable—turning prompt development into a scalable engineering practice.
::contentReference[oaicite:0]{index=0}



## 1. PDF Layout & Block Segmentation
- **PyMuPDF (fitz)** — for extracting text blocks, bounding boxes, page geometry. :contentReference[oaicite:1]{index=1}
- **Unstructured** (hi_res strategy) — for classifying blocks into categories (Table, Narrative, etc.) :contentReference[oaicite:2]{index=2}

## 2. Table Detection & Structure Extraction
- **Table Transformer (TATR)** — state-of-the-art table detection and structured output (HTML/CSV + geometry).
- **PyMuPDF’s native table layout detection** — lighter, good for simpler layouts. :contentReference[oaicite:3]{index=3}
- **Camelot** (stream/lattice modes) — handy for clear, single-page tables. :contentReference[oaicite:4]{index=4}

## 3. Multi-Page Table Stitching
- **Pandas & NumPy** — align columns by approximate x-coordinate.
- **Geometry-based merging** using PyMuPDF word/block boxes.

## 4. OCR on Table Crops (for scanned PDFs)
- **docTR (MINDee)** — modern, performant OCR for selective regions.
- **Tesseract** as fallback for compatibility.

## 5. Narrative Linking & Semantic Retrieval
- **LangChain** or **custom embeddings** + vector store client (e.g., pgvector, Qdrant) — retrieve narrative chunks by semantic similarity.

## 6. Apply Corrections / Normalization
- **Standard Python `re`** — detect numeric overrides like “revised to 1,234...”
- **LLM client (e.g., OpenAI / Claude)** for unit parsing, entity normalization, or edge-case disambiguation.

## 7. Evaluation and Backtesting
- **GriTS metric** from TATR / PubTables-1M repo — to assess table extraction structure/content. :contentReference[oaicite:5]{index=5}
- **Custom metrics** — precision/recall for narrative overrides, value error, latency.

## 8. Storage & Schema
- **SQLite** (local prototyping) or **PostgreSQL + pgvector** (scalable).
- Schema tables: `pdf_doc`, `blocks`, `tables`, `rows`, `corrections`.

---

### Optional and Advanced Tools
- **Unstructured** supports layout-aware document partitioning for RAG contexts. :contentReference[oaicite:6]{index=6}
- **PdfTable** toolkit — handles digital and image-based tables with multiple models. :contentReference[oaicite:7]{index=7}
- **Docling** — comprehensive conversion using layout and table models like DocLayNet and TableFormer. :contentReference[oaicite:8]{index=8}