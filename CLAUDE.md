# MarkShark - Claude Code Project Guide

## Project Overview
MarkShark is a bubble sheet grading system that processes scanned answer sheets, scores them against answer keys, and generates reports. It supports multiple interfaces: CLI, Streamlit web app, and PySide6 desktop GUI.

## Coding Philosophy — Long-Term Maintenance, Cross-Platform

MarkShark is academic software that must remain functional and maintainable for **5+ years** with minimal active development. It runs on both **macOS and Windows**. The original author has limited programming experience, so code should be readable by a competent programmer picking it up cold in 3 years on either platform.

### Priorities (in order)

1. **Cross-platform compatibility** — Always consider both Mac and Windows:
   - Use `pathlib.Path` instead of `os.path` (handles `/` vs `\` correctly)
   - Never hardcode path separators, drive letters, or case-sensitive filename assumptions
   - Use `\n` for internal strings; let Python's `open()` handle line-ending translation
   - Platform-specific GUI code must go through an abstraction (see `gui/utils.py`)
   - Test file dialogs and subprocess calls on both platforms

2. **Self-documenting comments** — Explain **why**, not what:
   - Why this approach was chosen over alternatives
   - What edge cases or platform quirks this handles
   - What would break if this code were changed
   - Comments should help a future maintainer, not narrate syntax

3. **Dependency awareness** — When adding or updating libraries:
   - Note what each dependency actually provides and why it was chosen
   - Verify it works on both Mac and Windows
   - Flag any OS-specific interfaces (these break first on OS updates)
   - Prefer stable, widely-used, cross-platform libraries over niche alternatives

4. **Testability** — Structure code so core functions can be tested independently:
   - Separate file I/O from processing logic
   - Keep scoring/parsing functions pure where possible (input → output, no side effects)
   - Document example inputs/outputs so the same test files verify both platforms

5. **Build documentation** — Keep PyInstaller build notes for both platforms:
   - Python version, OS version, any platform-specific flags
   - Where dependencies are specified (`pyproject.toml`)
   - Differences between Mac and Windows builds

### Avoid

- Platform-specific path assumptions (hardcoded slashes, drive letters)
- Clever one-liners that will be incomprehensible later
- Undocumented magic numbers or thresholds
- OS-specific GUI code without abstraction (use `gui/utils.py` helpers)
- Jargon-heavy explanations — define terms when they first appear

## Architecture

### Package Layout (`src/markshark/`)

The package root and `tools/` directory have distinct roles:

**Root level** — entry points, engines, and global config only:

| File | Role |
|---|---|
| `cli.py` | Typer-based CLI entry point (`align`, `score`, `report`, `gui`) |
| `align_core.py` | Alignment engine (`align_pdf_scans()`) |
| `score_core.py` | Scoring engine (`score_pdf()`) |
| `mapviewer_core.py` | Bubblemap overlay engine |
| `defaults.py` | Global config and scoring constants |
| `template_manager.py` | Core domain object — templates are central to the app |
| `mock_dataset.py` | Synthetic dataset generator for testing/demos |
| `app_streamlit.py` | Streamlit web interface |
| `gui/` | PySide6 desktop application (see GUI Structure below) |

**`tools/`** — reusable helper libraries with no CLI entry points:

| File | Called by | Purpose |
|---|---|---|
| `align_tools.py` | `align_core` | Image processing, ArUco detection, homography |
| `score_tools.py` | `score_core` | Bubble ROI scoring, grid centers, version detection |
| `key_parser.py` | `score_core`, `score_tools`, GUI key builder | Answer key parsing (text/CSV/Excel), scoring logic |
| `project_utils.py` | GUI pages | Project directory structure, archiving, metadata |
| `report_tools.py` | CLI, GUI | Excel report generation, item analysis |
| `stats_tools.py` | `report_tools` | Statistics computation |
| `bubblemap_io.py` | Multiple | Bubble sheet template I/O (YAML ↔ `Bubblemap`) |
| `io_pages.py` | `align_core`, `score_core` | PDF page loading/writing |
| `visualizer_tools.py` | `score_core` | Annotation rendering on images |

**Rule: where does a new module go?**
- If it defines a **CLI command** or is a **top-level processing engine** (input → output pipeline), it belongs at the **root**.
- If it is a **reusable helper library** called by engines or GUI pages (parsing, I/O, statistics, filesystem utilities), it belongs in **`tools/`**.
- When in doubt, put it in `tools/`. The root should stay small and easy to scan.

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
│   ├── key_builder.py      # Answer Key Utility (create, import, edit, export answer keys)
│   ├── pdf_tools.py        # PDF manipulation utilities
│   ├── lms_integration.py  # LMS gradebook import/export
│   ├── map_viewer.py       # Bubblemap Utility (bubblemap overlay viewer)
│   ├── mock_data_utility.py# Synthetic dataset generator
│   ├── project_manager_page.py # Course Manager (browse/manage courses & assessments)
│   ├── settings.py         # App settings
│   └── help_page.py        # Help & documentation
├── dialogs/
│   ├── about.py            # About dialog
│   └── course_dialog.py    # Create / edit / relocate course dialog
├── widgets/
│   ├── page_header.py
│   ├── pdf_preview.py      # PDF page display widget
│   └── project_selector.py
└── models/
    ├── project.py           # ProjectManager for directory/state
    ├── project_registry.py  # JSON-backed project + course registry (~/.markshark/)
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
| `create_new_project(parent_widget)` | Prompts for assessment name, creates directory structure (`input_files/`, `score_data/`, `logs/`), registers in `ProjectRegistry`. Returns `Path` or `None`. Use instead of duplicating the new-assessment flow. |

### Rules

1. **Never duplicate platform-branching logic** (`if Darwin … elif Windows … else xdg-open`). Always call `open_file_or_folder()`.
2. **Never duplicate the version-fetching try/except.** Always call `get_app_version()`.
3. **Never copy the run-button stylesheet literal** into a new file. Import `RUN_BUTTON_STYLE`.
4. **When copying bundled files** (templates, assets) always use `safe_copy_file()` so macOS hidden-flag issues don't resurface.
5. **If a new pattern appears in 2+ pages**, extract it to `utils.py` proactively rather than leaving the duplication for later.
6. **Import style**: Use lazy (in-method) imports for functions like `from ..utils import open_file_or_folder` to avoid circular-import risk. Use module-level imports for constants like `RUN_BUTTON_STYLE`.

## Terminology: UI vs Code

The UI uses teacher-friendly terms; internal code keeps programmer-friendly names:

| Teacher sees | Code uses | Meaning |
|---|---|---|
| **Course folder** | `working_dir`, `workdir` | Top-level directory for a class (e.g. `BIO101/`) |
| **Assessment** | `project`, `project_dir` | One test within a course (e.g. `midterm_1/`) |
| **Course Manager** | `ProjectManagerPage` | Page that browses/manages courses and assessments |

Settings keys (`"project/working_dir"`, `"project/last_project"`) and registry files (`projects.json`) keep their original names for backward compatibility.

### Registry Schema (v2)
`~/.markshark/projects.json` stores both **courses** (course folders) and **projects** (assessments) as sibling lists. v1 files are auto-migrated on first load. Key methods on `ProjectRegistry`:
- `register_course(path, name)` / `list_courses()` — manage known course folders
- `list_courses()` — returns courses sorted most-recently-active first (max of course + child assessment timestamps)
- `set_course_name(path, name)` — rename a course (display name only)
- `update_course_path(old, new)` — re-point a course + child assessments to a new folder
- `update_course_last_opened(path)` — touch a course's last_opened timestamp
- `list_by_course()` — group projects by parent course folder (orphans under `"__orphan__"`)
- `register(project_path)` — auto-registers the parent as a course

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

### Simple Grade Mode
A streamlined grading mode for small classes (no student IDs, no roster matching). Activated via a checkbox on the Quick Grade page.

- **Per-assessment flag**: `ProjectRegistry.set_simple_mode()` / `get_simple_mode()` — stored in `projects.json` alongside `template_id`
- **Global default**: `SettingsStore` key `"defaults/simple_grade"` (default: `False`)
- **Report**: `generate_report(..., simple=True)` produces Summary + Class Scores + Answer Key only (no per-version item analysis)
- **CLI**: `markshark report --simple` for the same streamlined output
- **Corrections**: In simple mode, corrections are keyed by **page number** (from the CSV `Page` column) instead of `StudentID`. The `merge_corrections()` function auto-falls back to Page matching when StudentID matching fails — no explicit mode flag needed.
- **Review panel**: Detects simple mode from ProjectRegistry on CSV load; uses page number as the correction key in `_on_cell_changed()`

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
| Add a new CLI command or root-level engine | Add it to the **Package Layout** root table |
| Add or move a module into `tools/` | Add it to the **Package Layout** tools table |
| Change the CSV output format | Update **CSV Output Format** |
| Change scoring/flagging conventions | Update **Key Patterns** |

### How to update
- Edit this file directly — keep entries concise (one line per item in tables/trees).
- The utils table should always match the actual exports in `gui/utils.py`. If they drift apart, the table is wrong — fix it.
- Commit the CLAUDE.md update in the **same commit** as the code change so they never get out of sync.
