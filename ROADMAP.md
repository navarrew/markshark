# MarkShark Roadmap

Planned features and improvements, roughly ordered by priority.

---

## Near-Term

### Mock Dataset: AND/OR Answer Keys
- Generate compound answer keys (`B&C` for AND, `B^C` for OR) at configurable rates
- Produce realistic student errors: partial fills for AND, multi-fills for OR
- Scoring engine already supports these modes — this is mock-data-only work
- **Status:** Implemented (claude/ecstatic-swartz branch)

### Mock Dataset: ID Mis-entry & Missing Version Errors
- Corrupt student IDs (typo, transposition, extra/missing digit)
- Skip version zone filling to simulate blank version fields
- Ground-truth CSV tracks OriginalID vs BubbledID
- **Status:** Implemented (claude/ecstatic-swartz branch)

### Map Viewer: Go-to-Page Spinbox
- Direct page jump via QSpinBox instead of sequential Prev/Next only
- **Status:** Implemented (claude/ecstatic-swartz branch)

---

## Medium-Term — New Modules

### Key Helper / Key Maker Utility
A dedicated GUI page to make it easy for teachers to create, import, and edit answer keys in MarkShark format. Two main workflows:

**Paste/Import:**
- Paste text from a Word doc, Google Sheets, etc. and auto-detect the answer format
- Parse common patterns: comma-separated (A,B,C,D...), one-per-line, numbered (1.A 2.B...), tab-separated
- Handle messy input gracefully (extra whitespace, mixed case, numbering prefixes)
- Output a clean MarkShark key file (.txt) with proper ver:/code: headers

**Manual Entry / Edit:**
- Grid-style editor: rows for Q1, Q2, ... with dropdown or button-group per question
- Support for advanced key types: AND (`B&C`), OR (`B^C`), freebie (`*`), custom points (`:3`)
- Load an existing key file to review, correct, or extend it
- Version management: add/remove versions, copy a version as starting point for another
- Save / export to MarkShark key format

**GUI design TBD** — could be a two-panel layout (paste area on left, structured editor on right) or a tabbed interface (Import tab / Editor tab). Needs experimentation to find what feels natural for teachers.

### Alignment Compare / Scan Browser
New GUI page for troubleshooting alignment problems, with two PDF panels side by side.
- Left panel: raw scans, Right panel: aligned scans
- Synchronized page navigation (shared go-to-page spinbox)
- Optional bubblemap overlay on aligned side (reuse `mapviewer_core.py` overlay logic)
- Helps teachers identify which pages aligned poorly and why
- Could also be useful for general QA of scan quality before scoring

### PDF Page Corrector
Lightweight utility for fixing individual mis-scanned pages before alignment. Simpler than a full image editor — just the operations teachers actually need:
- Page browser with PDF preview and go-to-page spinbox
- 180-degree flip (the most common fix — a page scanned upside-down)
- Rotation nudge (plus/minus 0.1-degree steps) for pages that are slightly skewed
- X/Y translation nudge for pages with unusual margins or feed offset
- Export corrected PDF (replace pages in-place or save as new file)
- Uses PyMuPDF or OpenCV for transforms

### Optional ID / Name-Only Mode
Allow teachers to skip ID-based matching and just get names + scores.
- Toggle in Quick Grade / Score Only pages
- Alternate code path in `score_core.py` that skips roster matching or matches by name
- Simplified report output (no orphan/absent classification)
- Use case: teachers who don't assign student IDs and just want name + score

---

## Longer-Term

### Survey Mode
New mode alongside grading — no answer key, no correct/incorrect.
- Collect response distributions per question
- Anonymous scan support (no roster required)
- Report output: frequency tables, bar charts per question
- Could be a new top-level tab ("Survey Mode") rather than living under "Grader"
- New `survey` CLI command, new GUI page, new report template
