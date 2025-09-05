"""
Self-coding preprocessor that lets the LLM write its own PDF extraction code.
This mirrors what ChatGPT does in the web interface with its augmented toolset.

Version: v1.5.0 - Two-stage self-coding system with script logging
"""

import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.llm_client import LLMClient

SYSTEM_MSG = """You are a senior Python engineer specializing in WHO surveillance bulletin data extraction. Analyze the uploaded WHO AFRO bulletin PDF to extract surveillance records.

EXPECTED PDF STRUCTURE (typical WHO bulletins):
- Main surveillance table usually starts around pages 8-10 with header "All events currently being monitored by WHO AFRO"
- Table spans multiple pages with consistent column structure, but Column names are only shown 1 time above the first table record. They are never repeated
- Columns: Country | Event | Grade | Date notified to WCO | Start/End reporting periods | Total cases | Cases Confirmed | Deaths | CFR
- Narrative descriptions appear directly BELOW each table row, we want to put this as a new column for the row.

YOUR TASK:
1. Accept PDF path via --pdf command line argument
2. Locate the main surveillance table (look for "All events currently being monitored" or similar headers)
3. Extract table data across multiple pages, handling page breaks correctly
4. Associate narrative text with table rows correctly
5. Output to result.json as JSON array

TECHNICAL APPROACH FOR PDFPLUMBER:
- Use page.extract_tables() to get table structure, but VALIDATE each row
- Real table rows have 10 columns and start with a country name (e.g., "Kenya", "Nigeria", "Cameroon")
- Narrative paragraphs get extracted as fake "table rows" with text in the first column only
- Use len(row) >= 8 and validate row[0] contains actual country names vs long descriptions
- For narrative text: Look for text blocks that appear between table rows using page.extract_text()
- Use regex/text analysis to find the narrative paragraph that follows each table row
- Country name validation: Use a list of African countries or check if row[0] is short (< 50 chars) vs long descriptions

TABLE COLUMNS TO EXTRACT:
- From table: Country, Event, Grade, Date notified to WCO, Start of reporting period, End of reporting period, Total cases, Cases Confirmed, Deaths, CFR
- Add manually: PageNumber (which page the record was found on), NarrativeText (descriptive text below each table row)


EXAMPLE - EXPECTED JSON OUTPUT STRUCTURE:
```json
[
  {
    "Country": "Kenya",
    "Event": "Cholera",
    "Grade": "Grade 3",
    "Date notified to WCO": "17-Feb-25",
    "Start of reporting period": "10-Feb-25", 
    "End of reporting period": "16-Jul-25",
    "Total cases": "423",
    "Cases Confirmed": "99",
    "Deaths": "20",
    "CFR": "4.7%",
    "PageNumber": 9,
    "NarrativeText": "As of 13 July 2025, Kenya has reported 423 cholera cases across five counties."
  }
]
```
"""


USER_PLAN_TMPL = """Analyze the attached PDF file and write a Python script named main.py that extracts surveillance data from it.

Requirements:
- Use argparse to accept --pdf command line argument
- Based on your analysis of the PDF structure, extract table data from the appropriate surveillance pages
- Output JSON array to result.json file
- Handle multi-page tables where headers are on first page and data continues
- Include narrative text blocks between table rows as a bew column for the row above (not a new row)
- Adapt your extraction strategy based on the actual PDF layout you observe

Write ONLY the Python code for main.py. Do not include any JSON wrapper or explanations."""


REFINE_TMPL = """The previous run failed or produced invalid output.

PDF: {pdf_path}

STDOUT:
{stdout}

STDERR:
{stderr}

Please fix your Python code and return ONLY the corrected main.py code."""


def _call_llm_with_pdf(prompt: str, client: LLMClient, pdf_path: str) -> str:
    """Call LLM with PDF upload and return Python code directly."""
    import base64

    system_prompt = SYSTEM_MSG
    user_prompt = prompt

    print(f"📤 Uploading PDF to LLM for analysis: {Path(pdf_path).name}")

    # Get model info for routing
    model_info = client.get_model_info()
    provider = model_info["provider"]
    actual_model_name = model_info["model_name"]

    # Use appropriate PDF upload method based on provider
    if provider == "openai":
        # OpenAI base64 inline upload
        try:
            with open(pdf_path, "rb") as pdf_file:
                pdf_b64 = base64.b64encode(pdf_file.read()).decode("utf-8")

            print(f"✅ PDF encoded successfully: {len(pdf_b64)} base64 characters")

            # Create message with PDF attachment
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:application/pdf;base64,{pdf_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ]

            response = client.client.chat.completions.create(
                model=actual_model_name,
                messages=messages,
                max_tokens=100000,
                temperature=0,
            )

            resp = response.choices[0].message.content

        except Exception as e:
            print(f"❌ OpenAI PDF upload failed: {e}")
            raise

    elif provider == "openrouter":
        # OpenRouter PDF upload via multipart
        try:
            # Use OpenRouter's file upload approach (simplified version)
            with open(pdf_path, "rb") as pdf_file:
                pdf_content = pdf_file.read()
                pdf_b64 = base64.b64encode(pdf_content).decode("utf-8")

            print(f"✅ PDF prepared for OpenRouter: {len(pdf_b64)} base64 characters")

            # Create message with PDF inline (OpenRouter supports this)
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:application/pdf;base64,{pdf_b64}"
                            },
                        },
                    ],
                },
            ]

            response = client.client.chat.completions.create(
                model=actual_model_name,
                messages=messages,
                max_tokens=100000,
                temperature=0,
            )

            resp = response.choices[0].message.content

        except Exception as e:
            print(f"❌ OpenRouter PDF upload failed: {e}")
            raise
    else:
        raise ValueError(f"Provider {provider} does not support PDF upload")

    # Extract Python code from response
    code = resp.strip()

    # Remove code block markers if present
    if "```python" in code:
        start = code.find("```python") + 9
        end = code.find("```", start)
        if end != -1:
            code = code[start:end].strip()
    elif "```" in code:
        start = code.find("```") + 3
        end = code.rfind("```")
        if end != -1 and end > start:
            code = code[start:end].strip()

    return code


def _call_llm(prompt: str, client: LLMClient, pdf_path: str = None) -> str:
    """Call LLM and return Python code directly. Uploads PDF if provided."""
    if pdf_path and Path(pdf_path).exists():
        return _call_llm_with_pdf(prompt, client, pdf_path)
    else:
        # Fallback to text-only
        print("⚠️ No PDF path provided or file doesn't exist, using text-only mode")
        system_prompt = SYSTEM_MSG
        user_prompt = prompt

        resp, meta = client.create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=100000,
            temperature=0,
        )

        # Extract Python code from response
        code = resp.strip()

        # Remove code block markers if present
        if "```python" in code:
            start = code.find("```python") + 9
            end = code.find("```", start)
            if end != -1:
                code = code[start:end].strip()
        elif "```" in code:
            start = code.find("```") + 3
            end = code.rfind("```")
            if end != -1 and end > start:
                code = code[start:end].strip()

        return code


def run_self_coding_preprocessor(pdf_path: str, max_iters: int = 3) -> Dict[str, Any]:
    """
    Let the LLM write its own preprocessing code and iteratively refine it.

    Args:
        pdf_path: Path to PDF file to process
        max_iters: Maximum refinement iterations

    Returns:
        Dict with extraction results including generated scripts

    Version: v1.5.0 - Added script logging for debugging and reuse
    """
    # Setup script logging directory
    from src.config import Config

    script_logs_dir = Path(Config.LOGS_DIR) / "generated_scripts"
    script_logs_dir.mkdir(parents=True, exist_ok=True)

    client = LLMClient()  # Uses your config/model routing

    print(f"🤖 Letting the model author preprocessing code for: {Path(pdf_path).name}")

    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)

        # Initial plan from LLM
        print("📝 Getting initial preprocessing plan from LLM...")
        code = _call_llm(USER_PLAN_TMPL, client, pdf_path=pdf_path)

        for attempt in range(1, max_iters + 1):
            print(f"🔄 Attempt {attempt}/{max_iters}")

            # Write the Python script
            main_py = tmp / "main.py"
            main_py.write_text(code, encoding="utf-8")
            print(f"✍️  Wrote main.py ({len(code)} chars)")

            # Install common packages
            packages = ["pdfplumber", "regex", "pandas", "PyPDF2"]
            for pkg in packages:
                print(f"📦 Installing {pkg}...")
                subprocess.run(
                    ["python", "-m", "pip", "install", "-q", pkg],
                    check=False,
                    capture_output=True,
                )

            # Run the generated code
            stdout, stderr = "", ""
            try:
                cmd = ["python", "main.py", "--pdf", pdf_path]
                print(f"🚀 Running: {' '.join(cmd)}")
                proc = subprocess.run(
                    cmd,
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    timeout=120,  # 2 minute timeout
                )
                stdout += proc.stdout
                stderr += proc.stderr

                if proc.returncode != 0:
                    raise RuntimeError(f"Command failed with code {proc.returncode}")

            except Exception as e:
                print(f"❌ Execution failed: {e}")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")

                # Ask LLM to refine the code
                if attempt < max_iters:
                    print("🔧 Asking LLM to fix the code...")
                    refine_prompt = REFINE_TMPL.format(
                        pdf_path=pdf_path, stdout=stdout, stderr=stderr
                    )
                    code = _call_llm(refine_prompt, client, pdf_path=pdf_path)
                    continue
                else:
                    return {
                        "success": False,
                        "error": f"Self-coding preprocessor failed after {max_iters} iterations: {e}",
                    }

            # Check if result.json was created
            result_file = tmp / "result.json"
            if result_file.exists():
                try:
                    # Read the JSON data for the second stage
                    raw_json_data = result_file.read_text(encoding="utf-8")
                    data = json.loads(raw_json_data)

                    print(
                        f"✅ Successfully extracted {len(data)} records via self-coded preprocessor"
                    )

                    # Check if we actually got meaningful data
                    if len(data) == 0:
                        print("⚠️ No records extracted - will try to refine the code")
                        if attempt < max_iters:
                            print("🔧 No data extracted, asking LLM to refine...")
                            refine_prompt = REFINE_TMPL.format(
                                pdf_path=pdf_path,
                                stdout=f"No records extracted. The script ran without errors but produced an empty result.json file.\n{stdout}",
                                stderr=stderr,
                            )
                            code = _call_llm(refine_prompt, client, pdf_path=pdf_path)
                            continue
                        else:
                            return {
                                "success": False,
                                "error": f"Self-coding preprocessor failed after {max_iters} iterations - no data extracted",
                            }

                    # Log successful scripts for debugging and reuse only if we got data
                    pdf_name = Path(pdf_path).stem
                    script_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    script_log_path = (
                        script_logs_dir / f"{pdf_name}_{script_timestamp}_main.py"
                    )
                    script_log_path.write_text(code, encoding="utf-8")
                    print(f"📁 Saved successful script: {script_log_path}")

                    # Save metadata about this successful run
                    metadata = {
                        "pdf_path": pdf_path,
                        "pdf_name": pdf_name,
                        "timestamp": script_timestamp,
                        "attempt": attempt,
                        "record_count": len(data),
                        "generated_files": ["main.py"],
                        "pip_packages": packages,
                        "run_commands": cmd,
                    }
                    metadata_path = (
                        script_logs_dir / f"{pdf_name}_{script_timestamp}_metadata.json"
                    )
                    metadata_path.write_text(
                        json.dumps(metadata, indent=2), encoding="utf-8"
                    )
                    print(f"📁 Saved script metadata: {metadata_path}")

                    return {
                        "success": True,
                        "raw_json_data": raw_json_data,
                        "record_count": len(data),
                        "preprocessing_log_id": None,
                        "attempt": attempt,
                        "generated_files": ["main.py"],
                        "script_log_dir": str(script_logs_dir),
                        "script_timestamp": script_timestamp,
                    }

                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON in result.json: {e}")

            # No valid result → refine
            if attempt < max_iters:
                print("🔧 No valid result.json found, asking LLM to refine...")
                refine_prompt = REFINE_TMPL.format(
                    pdf_path=pdf_path, stdout=stdout, stderr=stderr
                )
                code = _call_llm(refine_prompt, client, pdf_path=pdf_path)

        return {
            "success": False,
            "error": f"Self-coding preprocessor failed after {max_iters} iterations - no valid result.json",
        }
