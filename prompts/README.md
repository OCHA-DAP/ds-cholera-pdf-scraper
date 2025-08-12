# Prompt Management Structure

This directory contains prompt templates organized by type and format.

## Directory Structure

```
prompts/
├── markdown/                    # Human-editable markdown prompts
│   └── health_data_extraction/  # Prompt type
│       └── health_data_extraction_v1.0.0.md
├── health_data_extraction/      # Generated JSON prompts (auto-created)
│   └── health_data_extraction_v1.0.0.json
└── prompt_metadata.json        # Version tracking metadata
```

## Workflow

### 1. Edit Prompts (Markdown)
- Create/edit prompts in `markdown/{prompt_type}/` directory
- Use semantic versioning (v1.0.0, v1.0.1, v2.0.0, etc.)
- Include YAML frontmatter with version, description, and created_at

### 2. Import to System (JSON)
```bash
python prompt_cli.py import-from-markdown \
  --prompt-type health_data_extraction \
  --markdown-file prompts/markdown/health_data_extraction/health_data_extraction_v1.0.0.md
```

### 3. Set as Current Version
```bash
python prompt_cli.py set-current \
  --prompt-type health_data_extraction \
  --version v1.0.0
```

### 4. Export for Editing
```bash
python prompt_cli.py export-to-markdown \
  --prompt-type health_data_extraction \
  --version v1.0.0 \
  --output prompts/markdown/health_data_extraction/health_data_extraction_v1.1.0.md
```

## Benefits

- **Human-readable**: Edit prompts in markdown format
- **Version control**: Clean diffs in git
- **Automated logging**: JSON system handles execution tracking
- **Organized storage**: Clear separation of source vs generated files
