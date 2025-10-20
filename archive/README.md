# Archived Components

This directory preserves deprecated and experimental code for historical reference.

## deprecated/ (Committed Code, No Longer Active)

### extraction/
Alternative extraction approaches superseded by main pipeline:
- **llm_extract_hybrid.py**: Experimental hybrid LLM+rules approach
- **standalone_stage1_extractor.py**: Standalone extraction experiment

**Note**: `pre_extraction/who_surveillance_extractor.py` was initially considered for archiving but restored after database verification showed active usage by the `--preprocessor table-focused` flag.

### preprocessing/
Preprocessing modules not used by active --preprocessor flags:
- **pdf_segmentation.py**: PDF page segmentation (not integrated)
- **table_detection.py**: Superseded by table_detection_surveillance.py
- **table_stitching.py**: Table spanning logic (not integrated)
- **table_to_llm_format.py**: Table formatting (not used)
- **narrative_linking.py**: Superseded by llm_narrative_linking.py
- **surveillance_preprocessing_pipeline.py**: Pipeline (not integrated)
- **preprocessing_manager.py**: Manager class (not used)

### batch_processing/
Alternative batch processors superseded by batch_run_extraction.py:
- **simple_batch_processor.py**: Uses deprecated pre_extraction module
- **batch_wrangle_pdf_to_json.py**: Sophisticated rule-based approach with:
  - Spatial PDF coordinate analysis
  - 100+ country name standardization rules
  - Advanced narrative extraction algorithms
  - Valuable fallback if LLM costs become prohibitive

### utils/
Utility modules that may not be actively used:
- **raw_response_storage.py**: Raw LLM response logging

**Note**: `run_id_manager.py` and `tabular_preprocessing_logger.py` were initially archived but restored after database verification showed they are actively used by the `--preprocessor table-focused` flag for logging to the `tabular_preprocessing_logs` table.

### scripts/
One-off test and utility scripts:
- **batch_pdf_processor.py**: Empty file (0 bytes)
- **test_generated_script.py**: Self-coding preprocessor tests
- **organize_preprocessing_files.py**: File organization utility
- **create_tabular_preprocessing_table.py**: Preprocessing experiments

## exploration/ (Untracked Experimental Work)

### docs/
Markdown summaries of experiments and refactoring plans:
- CFR visualization summaries
- Narrative linking integration docs
- Preprocessor usage guides
- Refactoring documentation

### analysis/
Quarto notebooks and analysis scripts:
- Database exploration guides
- Narrative linking demos
- Table extraction debugging
- LLM correction pipeline experiments

### scripts/
One-off test and exploration scripts:
- Database relationship setup
- Country standardization tests
- Logging integration tests
- Batch processor experiments

### notebooks/
Jupyter notebooks:
- Model performance analysis

### tmp/
Temporary analysis outputs and extracted text

## 🔄 Resurrection Guide

To bring back any archived component:

1. **Review the code** - Ensure it's still relevant and compatible
2. **Check dependencies** - Update imports and dependencies
3. **Move it back**:
   ```bash
   git mv archive/deprecated/path/to/file.py src/path/to/file.py
   ```
4. **Update imports** - Fix any broken imports in active codebase
5. **Add tests** - If moving to production, add proper tests
6. **Document** - Update README and architecture docs

## 📊 Why These Were Archived

### Extraction Methods
Superseded by unified llm_text_extract.py + pdf_upload_extract.py pipeline

### Preprocessing Modules
Not connected to any active --preprocessor CLI flags

### Batch Processors
batch_run_extraction.py is the active batch processor (outputs to batch_run/)

### Utilities
May not be actively called by production pipeline (verify before deletion)

### Exploration Files
Temporary experiments and analysis, not meant for production codebase

## 💡 Key Insights

**batch_wrangle_pdf_to_json.py** represents significant engineering effort:
- Solved the "narrative linking" problem with pure rule-based approach
- 100+ country name variations handled
- Spatial coordinate analysis for precise extraction
- Could be resurrected if LLM costs become an issue

**The Evolution**:
1. Rule-based extraction (wrangle)
2. Hybrid approaches (llm_extract_hybrid, pre_extraction for table-focused)
3. Pure LLM with preprocessing options (current pipeline)

**Note**: `src/pre_extraction/who_surveillance_extractor.py` remains active as part of the `--preprocessor table-focused` option, which combines rule-based table extraction with LLM correction.

The LLM approach won due to:
- Better edge case handling
- Less maintenance (no massive mapping tables)
- More accurate narrative extraction
- Automatic adaptation to format changes

## 📅 Archive Date

Archived: October 2025

Branch: focus-preprocessing → feat/llm-extraction-pipeline

Commit: cae1cbe
