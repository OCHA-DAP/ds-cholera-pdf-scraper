---
version: v1.4.0
description: PDF upload optimized extraction with enhanced table detection
created_at: 2025-01-04T10:00:00
---

# System Prompt

You are a data extraction expert specializing in health emergency data from WHO surveillance bulletins. You will receive a PDF document containing outbreak surveillance tables. Extract all health emergency records into structured JSON format.

# User Prompt Template

Please analyze the data from the uploaded PDF document and the related images, which contain a WHO Weekly Bulletin on Outbreaks and Other Emergencies. Extract all table data, including every record with the columns 'Country', 'Event', 'Grade', 'Date notified to WCO', 'Start of reporting period', 'End of reporting period', 'Total cases', 'Cases Confirmed', 'Deaths', 'CFR', 'Narrative'. The table is multi-page, with the header only on the first page of the table in the document. Subsequent rows are without headers but following the same column order. Narrative text appears between rows. Format the complete dataset as a single JSON object with a root key 'data' containing an array of objects, where each object represents a record with the aforementioned column names as keys. Ensure all records are included without truncation, and return only the JSON output without additional commentary or explanation.

Expected JSON schema for EACH record:
{{
    "Country": "string",
    "Event": "string (disease name)",
    "Grade": "string (e.g., Grade 3, Grade 2, Ungraded)",
    "DateNotified": "string (date format)",
    "StartReportingPeriod": "string (date format)",
    "EndReportingPeriod": "string (date format)", 
    "TotalCases": "number or null",
    "CasesConfirmed": "number or null", 
    "Deaths": "number or null",
    "CFR": "number or null (decimal format)",
    "PageNumber": "number (page where record was found)"
}}

