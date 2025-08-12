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
