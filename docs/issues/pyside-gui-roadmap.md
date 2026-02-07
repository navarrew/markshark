# PySide6 GUI Roadmap for MarkShark

## Overview

A native desktop GUI alternative to the Streamlit app, designed for packaging as a standalone application for teachers. Built with PySide6 for cross-platform support (macOS, Windows, Linux).

**Current Status:** Basic Quick Grade window implemented (`src/markshark/quickgrade_gui.py`)

---

## Phase 1: Core Quick Grade (In Progress)

- [x] Basic window layout with tabs
- [x] Template selector (loads from TemplateManager)
- [x] File selectors for scans, key, roster
- [x] Scoring options (annotate, threshold, etc.)
- [x] Alignment options (method, DPI, markers)
- [x] Async CLI execution with QProcess
- [x] Live log output
- [x] Progress indicator
- [ ] Output file open buttons (implemented but untested)
- [ ] Report generation step

---

## Phase 2: Built-in Review & Corrections Panel

**Priority: HIGH** - This is the killer feature that makes PySide6 worth the effort over Streamlit.

### Description
Instead of exporting flagged items to Excel, having teachers edit offline, and re-uploading, provide an integrated review experience:

```
┌─────────────────────────────────────────────────────────────────┐
│  [< Prev]  Page 5 of 12 (flagged)  [Next >]    [Save & Close]   │
├───────────────────────────────────┬─────────────────────────────┤
│                                   │  Flagged Items on This Page │
│                                   │  ─────────────────────────  │
│    [Scanned page image with       │  Q12: Blank detected        │
│     highlighted problem areas]    │       ○A ○B ●C ○D ○E        │
│                                   │                             │
│                                   │  Q15: Ambiguous (A+B)       │
│                                   │       ●A ○B ○C ○D ○E        │
│                                   │                             │
│    (zoom/pan controls)            │  Q23: Low confidence        │
│                                   │       ○A ○B ○C ●D ○E        │
│                                   │                             │
└───────────────────────────────────┴─────────────────────────────┘
```

### Features
- [ ] PDF page viewer with zoom/pan (QGraphicsView)
- [ ] Highlight flagged bubble regions on the image
- [ ] Side panel showing flagged items for current page
- [ ] Radio buttons or clickable bubbles for corrections
- [ ] Prev/Next navigation through flagged pages only
- [ ] Skip already-reviewed pages
- [ ] Show original detected answer vs. correction
- [ ] Auto-save corrections to memory
- [ ] Apply corrections to CSV before report generation
- [ ] Keyboard shortcuts for quick navigation (arrow keys, number keys for answers)

### Technical Notes
- Use `pdf2image` or `PyMuPDF` to render PDF pages to QPixmap
- Parse `flagged_for_review.xlsx` or raw scoring data for flagged items
- Store corrections in dict: `{(page, question): corrected_answer}`
- Overlay graphics items on QGraphicsScene for bubble highlights

### Effort Estimate
1-2 days for basic implementation, additional time for polish

---

## Phase 3: Additional Features

### Results Viewer
- [ ] Display results.csv in a QTableView
- [ ] Sort/filter by score, student ID, flags
- [ ] Click row to jump to that student's scan
- [ ] Export filtered results

### Student Lookup
- [ ] Search by student ID or name
- [ ] Show all responses for a student
- [ ] Compare to answer key visually

### Batch Processing
- [ ] Queue multiple scan PDFs
- [ ] Progress for batch operations
- [ ] Summary report across batches

### Settings/Preferences
- [ ] Remember last used paths (QSettings)
- [ ] Default options persistence
- [ ] Theme selection (light/dark)

---

## Phase 4: Packaging & Distribution

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

**Key insight:** The class-based structure in PySide6 is more verbose upfront but enables complex features like the review panel that would be hacky in Streamlit.

---

## Running the Current GUI

```bash
# Activate environment with PySide6
conda activate bubblefish-gui

# Install markshark in dev mode if needed
pip install -e /path/to/markshark

# Run the GUI
python -m markshark.quickgrade_gui
```

---

## Related Files

- `src/markshark/quickgrade_gui.py` - Main Quick Grade window
- `src/markshark/testgui.py` - Minimal align-only test GUI
- `src/markshark/app_streamlit.py` - Streamlit app (reference implementation)
