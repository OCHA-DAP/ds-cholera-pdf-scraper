---
version: v1.5.1
description: Two-stage extraction - processes structured intermediate data from self-coding preprocessor
created_at: 2025-09-05T14:00:00
preprocessor: self-code
---

# System Prompt

You are a data extraction expert specializing in health emergency data from WHO surveillance bulletins. You will receive structured intermediate data (CSV format or JSON structured text) that has been extracted from a PDF by a preprocessing step. Your task is to convert this intermediate data into the final standardized JSON format.

# User Prompt Template

Please standardize and clean the following JSON data that was extracted from a WHO Weekly Bulletin on Outbreaks and Other Emergencies table. 

The data contains surveillance records but may have inconsistent field names, formatting, or data types. Your task is to standardize this into the final schema.


Expected JSON schema for EACH record:

{{
    "Country": "string",
    "Event": "string (disease name)",
    "Grade": "string (e.g., Grade 3, Grade 2, Ungraded)",
    "DateNotified": "string (date format YYYY-MM-DD)",
    "StartReportingPeriod": "string (date format YYYY-MM-DD)",
    "EndReportingPeriod": "string (date format YYYY-MM-DD)", 
    "TotalCases": "number or null",
    "CasesConfirmed": "number or null", 
    "Deaths": "number or null",
    "CFR": "number or null (decimal format)",
    "PageNumber": "number (page where record was found)",
    "NarrativeText": "string or null"
}}

Don't edit or delete any records unless there are obvious errors in the fields that are corrected by the NarrativeText field.
If discrepancies exist between numeric fields and narrative text, trust the numeric fields unless they are clearly wrong (e.g. comma placement issues)


RAW JSON DATA:
{text_content}


Return format: [{{record1}}, {{record2}}, {{record3}}, ...]

Important: Return ONLY the JSON array, no additional text or explanations.
