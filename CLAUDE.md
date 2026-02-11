# MarkShark - Claude Code Project Guide

## Project Overview
MarkShark is a bubble sheet grading system that processes scanned answer sheets, scores them against answer keys, and generates reports. It supports multiple interfaces: CLI, Streamlit web app, and PySide6 desktop GUI.

## Architecture

### Core Components
- **`score_core.py`** - Main grading engine. Outputs simplified CSV with columns: `Page, Version, LastName, FirstName, StudentID, Correct, Incorrect, Blank, Multi, Flagged, FlagDetails, Q1-Qn`
- **`cli.py`** - Typer-based CLI with commands: `align`, `score`, `report`, `gui`
- **`app_streamlit.py`** - Streamlit web interface
- **`gui/`** - PySide6 desktop application (in active development)

### CSV Output Format (New Simplified Format)
```csv
Page,Version,LastName,FirstName,StudentID,Correct,Incorrect,Blank,Multi,Flagged,FlagDetails,Q1,Q2,...
KEY,A,KEY,KEY,KEY,,,,,,,A,B,...
1,A,Smith,John,12345,45,2,3,0,Y,Q5:blank|Q10:multi,A,B,...
```
- `Page` references annotated PDF page numbers
- `Flagged` is "Y" or blank
- `FlagDetails` uses pipe-separated codes: `Q5:blank|Q10:multi|ID:orphan`
- Stats/reports are generated separately (not in this CSV)

### GUI Structure (`src/markshark/gui/`)
```
gui/
├── app.py              # Main application entry
├── main_window.py      # Main window with sidebar navigation
├── pages/
│   ├── quick_grade.py  # Align & Score workflow
│   ├── review_panel.py # Review flagged items, apply corrections
│   └── template_manager.py
├── widgets/
│   ├── page_header.py
│   ├── pdf_preview.py  # PDF page display widget
│   └── project_selector.py
└── models/
    ├── project.py      # ProjectManager for directory/state
    └── corrections.py  # CorrectionLog append-only log
```

## Key Patterns

### Field Detection in CSV
Use `_get_field()` helper to handle multiple column name variants:
```python
score = _get_field(row_data, "Correct", "correct", "Percent", "percent")
```

### Flagging Format
- Old format: `Flagged` column with "Q3|Q15"
- New format: `FlagDetails` column with "Q5:blank|Q10:multi|ID:orphan"

### PDF Preview
`PDFPreview` widget caches by (path, page). Call `load_pdf(path, page=N)` where N is 0-indexed.

## Current Feature Branch Focus
This branch (`feature/gui-simplified-csv`) focuses on:
1. Simplified CSV output from `score_core.py` (no stats, no fancy headers)
2. Review & Correct panel with PDF page preview
3. Horizontal scrolling in student table
4. Page column linking CSV rows to annotated PDF pages

## Testing
```bash
# Run scoring
markshark score aligned.pdf --bublmap template.yaml --key-txt key.txt --out-csv results.csv

# Launch GUI
markshark gui
```

## File Naming Conventions
- Scored CSV: `results.csv`, `scored_*.csv`
- Corrections: `corrections_*.csv` (append-only log)
- Annotated PDF: `scored_scans.pdf`, `scored.pdf`
