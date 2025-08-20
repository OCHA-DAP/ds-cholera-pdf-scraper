---
version: v1.1.0
description: Simplified comprehensive extraction - extract ALL records from document
created_at: 2025-08-11T14:00:00
---

# System Prompt

You are a data extraction expert. Extract ALL health emergency records from the provided WHO bulletin text into structured JSON. Your goal is comprehensive extraction - do not miss any records.

# User Prompt Template

Extract ALL health emergency records from this WHO emergency bulletin text.

EXTRACTION REQUIREMENTS:
- Extract EVERY country-event combination mentioned in the document
- Process ALL pages and sections thoroughly
- Include records from both tables and descriptive text
- Do not skip any entries, even if data is incomplete

FIELD EXTRACTION PRIORITIES:
- **CasesConfirmed**: Look in tables first, then descriptive text. This field is critical.
- **TotalCases**: Use table values, but if table shows truncated numbers (like "27,16"), check text for full numbers (like "27,160")
- **Deaths, CFR, Grade**: Extract from tables when available
- **Dates**: Extract notification dates and reporting periods
- **PageNumber**: Note which page each record appears on

Expected JSON schema for EACH record:
{{
    "Country": "string",
    "Event": "string (disease name like Cholera, Mpox, Measles, etc.)",
    "Grade": "string (e.g., Grade 3, Grade 2, Ungraded)",
    "DateNotified": "string (date when WHO was notified)",
    "StartReportingPeriod": "string",
    "EndReportingPeriod": "string", 
    "TotalCases": "number (prefer table, but use text if table appears truncated)",
    "CasesConfirmed": "number (CRITICAL FIELD - check tables and text)",
    "Deaths": "number",
    "CFR": "number (case fatality rate as number, not percentage string)",
    "PageNumber": "number (if identifiable from text)"
}}

IMPORTANT: Process the ENTIRE document. Extract ALL records. Be comprehensive - the goal is to extract every single health emergency mentioned in the document, even if some fields are missing data.

Return ONLY a JSON array containing ALL extracted records. No explanations.

TEXT TO PROCESS:
{text_content}
