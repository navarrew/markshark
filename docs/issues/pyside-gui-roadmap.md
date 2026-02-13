# PySide6 GUI Roadmap for MarkShark

## Overview

A native desktop GUI alternative to the Streamlit app, designed for packaging as a standalone application for teachers. Built with PySide6 for cross-platform support (macOS, Windows, Linux).

**Current Status:** Modular GUI structure implemented in `src/markshark/gui/`

---

## Architecture

### Project Management System

The GUI uses a structured project folder system:

```
project_folder/
├── inputs/                     # IMMUTABLE SOURCE FILES
│   ├── scans/
│   │   ├── active/             # Current scan set
│   │   └── archive/            # Previous uploads
│   ├── keys/                   # Answer keys (versioned)
│   └── rosters/                # Class rosters (versioned)
├── outputs/                    # MUTABLE WORK PRODUCTS
│   ├── aligned/
│   │   ├── run_001/            # Alignment runs
│   │   └── archive/
│   ├── scored_001.csv          # Scoring results (indexed)
│   ├── corrections_001.csv     # Append-only corrections log
│   └── archive/
└── logs/
    ├── project.json            # Master project state
    └── *.log                   # Processing logs
```

### Key Design Principles

1. **Inputs are immutable** - Scans, keys, rosters versioned with index
2. **Outputs are regenerable** - Can re-align, re-score from inputs
3. **Corrections are append-only** - Full history, easy revert
4. **Runs are indexed** - `001`, `002` - human readable in filenames
5. **Timestamps in metadata only** - Inside JSON/CSV headers, not filenames
6. **Active pointers in project.json** - GUI knows what's "current"

---

## Phase 1: Core Structure ✅ COMPLETE

- [x] Modular GUI structure in `src/markshark/gui/`
- [x] Main window with sidebar navigation
- [x] Page-based architecture (QStackedWidget)
- [x] Page header widget with icon
- [x] Project/directory selector with QSettings persistence
- [x] PDF preview widget (pdf2image/PyMuPDF)
- [x] Template Manager page
- [x] Settings page (placeholder)

---

## Phase 2: Project Management ✅ COMPLETE

- [x] ProjectManager class (`gui/models/project.py`)
  - [x] Project creation with standard folder structure
  - [x] Input versioning (scans, keys, rosters)
  - [x] Alignment run tracking
  - [x] Scoring run tracking
  - [x] Archive/restore functionality
  - [x] ZIP export for archiving

- [x] CorrectionLog class (`gui/models/corrections.py`)
  - [x] Append-only corrections log
  - [x] Support for answer, ID, name corrections
  - [x] Revert functionality
  - [x] Apply corrections to data
  - [x] Copy corrections between runs

---

## Phase 3: Review & Correct Panel ✅ COMPLETE

The "killer feature" - integrated review without Excel round-trips.

- [x] Student table with scores and flag indicators
- [x] Filter: Show All / Flagged Only toggle
- [x] Answer grid showing all answers for selected student
- [x] Color coding: normal (white), flagged (orange), corrected (blue)
- [x] One-click correction buttons (A-E, Skip, No Mark)
- [x] Question detail panel with original/corrected display
- [x] Prev/Next navigation through students
- [x] Revert to original button
- [x] Export final grades with corrections applied
- [x] View corrections log
- [x] Scan preview area (placeholder for image display)

### TODO for Review Panel
- [ ] Load actual scan images (currently placeholder)
- [ ] Highlight bubble regions on scan
- [ ] Keyboard shortcuts (arrow keys, A-E for answers)
- [ ] Jump to specific flagged question
- [ ] Student ID/Name correction UI

---

## Phase 4: Quick Grade Integration 🚧 IN PROGRESS

- [x] Basic Quick Grade page structure
- [ ] Integration with ProjectManager
- [ ] Upload handling with replace/add prompts
- [ ] Run selection (create new vs replace)
- [ ] Downstream impact warnings
- [ ] Connection to Review & Correct after scoring

---

## Phase 5: Additional Features

### Project File Manager (Future)
- [ ] Visual tree view of project structure
- [ ] File size breakdown
- [ ] Archive/restore/delete operations
- [ ] ZIP export with options
- [ ] File lineage visualization

### Results Viewer
- [ ] Display results.csv in a QTableView
- [ ] Sort/filter by score, student ID, flags
- [ ] Click row to jump to that student's scan

### Settings/Preferences
- [x] Remember last used paths (QSettings)
- [ ] Default options persistence
- [ ] Theme selection (light/dark)

---

## Phase 6: Packaging & Distribution

- [ ] PyInstaller configuration for standalone .app (macOS)
- [ ] PyInstaller configuration for .exe (Windows)
- [ ] Code signing (macOS notarization, Windows signing)
- [ ] Auto-update mechanism
- [ ] Installer for Windows (NSIS or similar)
- [ ] DMG creation for macOS

### Dependencies to Bundle
- PySide6
- OpenCV (for ArUco detection)
- pdf2image + poppler
- All markshark dependencies

---

## File Structure

```
src/markshark/gui/
├── __init__.py
├── __main__.py
├── app.py                      # QApplication setup
├── main_window.py              # Main window with sidebar
├── widgets/
│   ├── __init__.py
│   ├── page_header.py          # Icon + title + description
│   ├── project_selector.py     # Project/directory picker
│   ├── pdf_preview.py          # PDF rendering widget
│   ├── file_selector.py
│   └── log_viewer.py
├── pages/
│   ├── __init__.py
│   ├── quick_grade.py
│   ├── review_panel.py         # Review & Correct
│   ├── template_manager.py
│   ├── mock_data_utility.py
│   └── settings.py
├── models/
│   ├── __init__.py
│   ├── project.py              # ProjectManager, ProjectState
│   └── corrections.py          # CorrectionLog, Correction
├── workers/
│   ├── __init__.py
│   └── cli_runner.py           # QProcess async CLI execution
├── dialogs/
│   ├── __init__.py
│   └── about.py
└── resources/
    ├── __init__.py
    └── icons/
        └── SHARKICON.png
```

---

## Running the GUI

```bash
# Activate environment with PySide6
conda activate markshark-gui

# Run the GUI (entry point)
python -m markshark.gui

# Or via installed command
markshark-pyside  # (if entry point is configured)
```

---

## Notes on PySide6 vs Streamlit

| Aspect | Streamlit | PySide6 |
|--------|-----------|---------|
| Learning curve | Low (scripty) | Medium (OOP/classes) |
| Layout | Auto (vertical) | Manual (explicit layouts) |
| State management | Session state magic | Instance variables |
| Async operations | Tricky | Native (QProcess, signals) |
| File dialogs | Upload/download | Native OS dialogs |
| Packaging | Requires server | Standalone executable |
| Offline use | No | Yes |
| Native feel | Browser-based | True native widgets |

**Key insight:** The class-based structure in PySide6 is more verbose upfront but enables complex features like the review panel and project management that would be hacky in Streamlit.

---

## Future Goals

### Welcome Page (New Sidebar Landing Page)
- [ ] Create a friendly, cartoony-styled welcome page as the first item in the left sidebar
- [ ] Basic info about MarkShark — what it does, who it's for
- [ ] Big friendly buttons / illustrated instructions in a casual, approachable format
- [ ] "Download a Template PDF" button so users can grab a bubble sheet without navigating to the Template Manager
- [ ] Core message front and center: "All you need is your bubble sheet and your key! If you add a roster we can do even more!"
- [ ] Quick-start walkthrough or visual guide (scan → score → review)
- [ ] Links/buttons to jump to Grader, Template Manager, etc. for common first actions

### Default / Pinned Template per Project
- [ ] Allow a template to be "pinned" to a project so it auto-populates whenever that project is opened
- [ ] Global default template setting — user picks a favorite that pre-fills on all new projects
- [ ] Per-project override: project-level pin takes precedence over the global default
- [ ] Template selector still available for one-off changes, but the default removes repetitive re-selection
- [ ] Most teachers will find one template and reuse it almost exclusively; only central scanning services need to juggle multiple templates regularly

### Online Template Repository ("Template Store")
- [ ] Host a curated collection of templates on Google Drive, Dropbox, or a simple static web host
- [ ] Manifest file (JSON/YAML) at a known URL listing each template: name, description, thumbnail URL, PDF download URL, bubble map download URL, metadata (# of questions, # of versions, page size, etc.)
- [ ] In-app "Template Store" page or dialog that fetches the manifest and displays a browsable gallery with thumbnails and descriptions
- [ ] One-click download: user picks a template, app downloads the PDF + bubble map YAML into their local templates folder
- [ ] Search/filter by number of questions, page size, features (ID field, name field, version bubbles, etc.)
- [ ] Cache the manifest locally so the gallery loads instantly and only refreshes on demand or periodically
- [ ] Versioning: manifest includes a version per template so the app can show "update available" badges
- [ ] Works offline after initial download — templates are fully local once fetched
- [ ] Could start with Google Drive shared folder (public links) and migrate to a proper CDN/GitHub Pages later if needed

### QR Code on Templates (Auto-ID + Alignment)
- [ ] Print a QR code on each template that encodes the template name/ID and bubble map version
- [ ] Scanner reads the QR code and auto-selects the correct bubble map — no manual template selection needed
- [ ] QR code doubles as an alignment anchor: OpenCV's QR detector returns corner coordinates, providing position + rotation like ArUco markers
- [ ] Could replace one ArUco marker or supplement existing markers for redundancy
- [ ] Pairs naturally with the "pinned template" feature — QR makes it fully automatic even without pinning
- [ ] Graceful fallback: if QR detection fails (damaged, low-res scan), fall back to ArUco-only alignment

### Annotated PDF Score Overlay
- [ ] After scoring, stamp each page of the annotated PDF with a text block showing: student name, ID, score, percentage, date scored
- [ ] Design templates with a dedicated "info strip" (top or bottom margin) intentionally left blank for this overlay
- [ ] Teacher can glance at any scanned page and immediately see results without cross-referencing the CSV
- [ ] Semi-transparent or margin-area overlay as an alternative for templates without a dedicated zone
- [ ] Optional: include flag summary (e.g., "2 flagged questions") so flagged sheets stand out at a glance

### PDF Report Output (Alternative to XLSX)
- [ ] Generate a polished PDF report mirroring the Excel report structure
- [ ] Page 1: Summary page — overall exam statistics and per-version stats table (like the Summary tab in XLSX)
- [ ] Per-version pages: item analysis, key, value row, % correct, point-biserial, item quality — one page per version
- [ ] Student scores table: paginated (e.g., ~40 students per page) so large classes flow across multiple pages
- [ ] Easier for teachers who just want to read/print results without needing Excel
- [ ] XLSX remains the primary format for teachers who want to copy/paste into their own gradebooks
- [ ] Could offer both formats from the same "Generate Report" button (checkboxes or dropdown)
