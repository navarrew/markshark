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
├── utils.py            # ★ Shared helpers — SEE "DRY / One Source of Truth" below
├── pages/
│   ├── welcome_page.py     # Default landing page
│   ├── quick_grade.py      # Align & Score workflow
│   ├── review_panel.py     # Review flagged items, apply corrections
│   ├── template_manager.py # Browse / manage bubble sheet templates
│   ├── align_only.py       # Alignment-only workflow
│   ├── score_only.py       # Score-only workflow
│   ├── report_only.py      # Report generation
│   ├── key_builder.py      # Key Build Utility (coming soon)
│   ├── pdf_tools.py        # PDF manipulation utilities
│   ├── lms_integration.py  # LMS gradebook import/export
│   ├── map_viewer.py       # Bubblemap overlay viewer
│   ├── mock_data_utility.py# Synthetic dataset generator
│   ├── project_manager_page.py # Project browser
│   ├── settings.py         # App settings
│   └── help_page.py        # Help & documentation
├── dialogs/
│   └── about.py            # About dialog
├── widgets/
│   ├── page_header.py
│   ├── pdf_preview.py      # PDF page display widget
│   └── project_selector.py
└── models/
    ├── project.py           # ProjectManager for directory/state
    ├── project_registry.py  # JSON-backed project registry (~/.markshark/)
    ├── corrections.py       # CorrectionLog append-only log
    └── lms_filter_registry.py
```

## DRY / One Source of Truth

**Before writing any new helper, check `gui/utils.py` first.** If a utility already exists there, import and reuse it. If you're about to write something that two or more pages will need, put it in `utils.py` instead of duplicating it.

### What lives in `gui/utils.py`

| Export | Purpose |
|---|---|
| `open_file_or_folder(path)` | Cross-platform open (macOS `open`, Windows `startfile`, Linux `xdg-open`). Use this instead of writing `platform.system()` / `subprocess.Popen` branching. |
| `safe_copy_file(src, dest)` | `shutil.copy` + strip macOS hidden/quarantine flags. Use when copying bundled assets (templates, etc.) so files appear in Finder. |
| `get_app_version()` | Returns `__version__` or `"development"`. Use instead of the `try: from markshark import __version__` boilerplate. |
| `RUN_BUTTON_STYLE` | Blue QPushButton stylesheet for prominent action buttons. Import as `from ..utils import RUN_BUTTON_STYLE`. |
| `TEAL`, `TEAL_HOVER`, `BLUE`, `BLUE_HOVER`, `GRAY_DISABLED` | Brand colour hex strings. Reference these instead of hard-coding hex values. |
| `template_display_label(template, tm)` | Returns display name with `★` prefix for favorites. Use when populating template combo boxes instead of raw `t.display_name`. |
| `create_new_project(parent_widget)` | Prompts for project name, creates directory structure (`input_files/`, `score_data/`, `logs/`), registers in `ProjectRegistry`. Returns `Path` or `None`. Use instead of duplicating the new-project flow. |

### Rules

1. **Never duplicate platform-branching logic** (`if Darwin … elif Windows … else xdg-open`). Always call `open_file_or_folder()`.
2. **Never duplicate the version-fetching try/except.** Always call `get_app_version()`.
3. **Never copy the run-button stylesheet literal** into a new file. Import `RUN_BUTTON_STYLE`.
4. **When copying bundled files** (templates, assets) always use `safe_copy_file()` so macOS hidden-flag issues don't resurface.
5. **If a new pattern appears in 2+ pages**, extract it to `utils.py` proactively rather than leaving the duplication for later.
6. **Import style**: Use lazy (in-method) imports for functions like `from ..utils import open_file_or_folder` to avoid circular-import risk. Use module-level imports for constants like `RUN_BUTTON_STYLE`.

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

## Keeping This File Up to Date

**CLAUDE.md is the project's source of truth for conventions.** It only works if it stays current. Update it as part of the same commit whenever you make a structural change. Specifically:

### When to update CLAUDE.md

| If you… | Then update… |
|---|---|
| Add a new page to `gui/pages/` | Add it to the **GUI Structure** tree |
| Add a new widget, model, or dialog | Add it to the **GUI Structure** tree |
| Add a new export to `gui/utils.py` | Add a row to the **What lives in `gui/utils.py`** table |
| Add a new DRY rule or convention | Add it to the **Rules** list |
| Add a new CLI command | Add it to **Core Components** |
| Change the CSV output format | Update **CSV Output Format** |
| Change scoring/flagging conventions | Update **Key Patterns** |

### How to update
- Edit this file directly — keep entries concise (one line per item in tables/trees).
- The utils table should always match the actual exports in `gui/utils.py`. If they drift apart, the table is wrong — fix it.
- Commit the CLAUDE.md update in the **same commit** as the code change so they never get out of sync.
