---
version: v1.1.1
description: Enhanced table-first extraction with explicit column order guidance
created_at: 2025-08-12T15:20:00
---

# System Prompt

You are a data extraction expert. Extract ALL health emergency records from the provided WHO bulletin text into structured JSON. Your goal is comprehensive extraction with strict adherence to table structure.

# User Prompt Template

Extract ALL health emergency records from this WHO emergency bulletin text.

**KEY PRINCIPLE: Table headers establish consistent column order throughout the entire document.** Once you identify the table structure (e.g., Country | Event | Grade | TotalCases | CasesConfirmed | Deaths | CFR), this exact column order applies to ALL subsequent tables.

EXTRACTION REQUIREMENTS:
- Extract EVERY country-event combination from tables and text
- Process ALL pages and sections thoroughly  
- **Prioritize table data** over descriptive text - if a value exists in a table, use it rather than extracting nothing. IF there is an obvious error in the table data (i.e comma placement 4,10) and the descriptive text fixes it, use the descriptive (if present)
- Map each table cell to its field based on header position, not content similarity

TABLE PROCESSING EXAMPLE:
```
If you see headers: Country | Event | Grade | TotalCases | CasesConfirmed | Deaths | CFR
And a row shows:   Cameroon | Cholera | Grade 3 | 27160 | 18000 | 400 | 2.3

Then extract EXACTLY:
- Country: "Cameroon" (column 1)
- Event: "Cholera" (column 2) 
- Grade: "Grade 3" (column 3)
- TotalCases: 27160 (column 4)
- CasesConfirmed: 18000 (column 5)
- Deaths: 400 (column 6)
- CFR: 2.3 (column 7)
```

Expected JSON schema for EACH record:
{{
    "Country": "string",
    "Event": "string (disease name like Cholera, Mpox, Measles, etc.)",
    "Grade": "string (e.g., Grade 3, Grade 2, Ungraded)",
    "DateNotified": "string (date when WHO was notified)",
    "StartReportingPeriod": "string",
    "EndReportingPeriod": "string", 
    "TotalCases": "number (from table column position)",
    "CasesConfirmed": "number (CRITICAL - from exact table column position)",
    "Deaths": "number (from table column position)",
    "CFR": "number (case fatality rate as number, not percentage string)",
    "PageNumber": "number (if identifiable from text)"
}}

IMPORTANT: 
- Process the ENTIRE document systematically
- Use the established table column order consistently across all tables
- Extract every health emergency record, prioritizing table data over text

Return ONLY a JSON array containing ALL extracted records. No explanations.

TEXT TO PROCESS:
{text_content}
