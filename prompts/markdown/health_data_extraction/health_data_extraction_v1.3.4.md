---
version: v1.3.4
description: Ultra-simple numerical corrections with forced JSON output
created_at: 2025-09-06T15:30:00
preprocessor: json-correction
---

# System Prompt

Fix only obvious number errors where narrative text contradicts table numbers. Respond only with valid JSON.

# User Prompt Template

Find numerical contradictions between table data and narrative text. Fix only obvious errors.

Return JSON in this exact format:

```json
{
  "corrections": [],
  "summary": {
    "total_records_reviewed": 0,
    "corrections_made": 0,
    "high_confidence_corrections": 0
  }
}
```

If you find obvious numerical errors, add them to corrections array with:
- record_index: array position number
- field: exact field name  
- old_value: exact current value
- new_value: corrected value from narrative
- confidence: 0.90-1.00
- explanation: brief reason

DO NOT change disease names. DO NOT add missing values. ONLY fix obvious wrong numbers.

Input data: $extracted_data