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

SYSTEM_MSG = """You are a senior Python engineer. Write a Python script to extract surveillance data from WHO AFRO bulletin PDFs.

Your script should:
1. Accept a PDF file path as a command line argument (--pdf)
2. Extract surveillance table data from pages 8-20 (or detect table pages automatically)
3. Extract these exact columns: Country, Event, Grade, Date notified to WCO, Start of reporting period, End of reporting period, Total cases, Cases Confirmed, Deaths, CFR, PageNumber, NarrativeText
4. Output the results to a file called result.json as a JSON array of records

Use libraries like pdfplumber, PyPDF2, pandas, regex as needed. Focus on robust extraction - don't worry about perfect formatting as that will be handled later."""


USER_PLAN_TMPL = """Write a Python script named main.py that extracts surveillance data from this PDF: {pdf_path}

Requirements:
- Use argparse to accept --pdf command line argument
- Extract table data from surveillance pages (usually pages 8-20)
- Output JSON array to result.json file
- Handle multi-page tables where headers are on first page and data continues
- Include narrative text blocks between table rows

Write ONLY the Python code for main.py. Do not include any JSON wrapper."""


REFINE_TMPL = """The previous run failed or produced invalid output.

PDF: {pdf_path}

STDOUT:
{stdout}

STDERR:
{stderr}

Please fix your Python code and return ONLY the corrected main.py code."""


def _call_llm(prompt: str, client: LLMClient) -> str:
    """Call LLM and return Python code directly."""
    system_prompt = SYSTEM_MSG
    user_prompt = prompt

    # Use the client's create_chat_completion method
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
        code = _call_llm(USER_PLAN_TMPL.format(pdf_path=pdf_path), client)

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
                    code = _call_llm(refine_prompt, client)
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
                            code = _call_llm(refine_prompt, client)
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
                code = _call_llm(refine_prompt, client)

        return {
            "success": False,
            "error": f"Self-coding preprocessor failed after {max_iters} iterations - no valid result.json",
        }
