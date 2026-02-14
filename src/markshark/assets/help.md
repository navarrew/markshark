# MarkShark Help
MarkShark is an optical mark recognition (OMR) system for bubble sheet grading.

---

## Getting Started

### Typical Workflow
1. **Scan** your bubble sheets to a single PDF (use a document scanner or scanning app).
2. **Open MarkShark** and go to the **Grader** page.
3. **Select your assessment** (or create a new one).
4. **Load your scans**, answer key, and optionally a roster.
5. Click **Grade** - MarkShark will align, score, and generate results.
6. **Review & Correct** flagged items on the Review page.
7. **Generate a report** with statistics, item analysis, and per-student details.

### File Structure and Assessments
MarkShark groups files together into assessments.  Typically an assessment is a single test (like 'BIO101 2025 midterm 1').  You can have a course folder for a class (like BIO101) with multiple assessments inside (midterm 1, midterm 2, final exam, etc.).
When you create a new assessment you typically upload your scanned student bubblesheets, your test key, and (optionally) a class roster.

```
my_assessment/          # example 'BIO101 midterm 1'
  input/                # where your original uploads are stored
    raw_scans.pdf       #
  runs/
    run_001/            # each grading run
      results.csv       # a raw output of student scores
      corrections.csv     #
      scored.pdf        #
  logs/           # correction logs, alignment logs
```

---

## Pages

### Grader
The main grading page. Load scans, select a template, provide an answer key and optional roster, then click Grade. Results appear in the output panel and you can generate an Excel report.

### Review & Correct
View scored results in a spreadsheet with the scanned PDF side-by-side. Click column headers to sort. Double-click answer cells to correct them via dropdown. Right-click corrected cells to revert. Corrections are saved to an append-only CSV log.

### Template Manager
Browse, preview, favourite, and reorder your installed bubble sheet templates. Each template includes a PDF master and a bubblemap YAML that defines bubble positions.

### Course Manager
Manage your courses and assessments, view contents, and open assessments directly in the Grader.

---

## Utilities

### Align Only
Run the alignment step independently - useful for troubleshooting alignment issues or preparing scans for external scoring.

### Score Only
Run scoring on pre-aligned PDFs without re-running alignment.

### Report Only
Generate a report from existing results and corrections without re-running the full pipeline.

### Mock Data Utility
Generate synthetic student datasets for testing. Creates fake scans, answer keys, response CSVs, and rosters from any installed template. Useful for testing templates before real grading.

### Map Viewer
Overlay bubblemap circles on a template PDF to visually verify that bubble positions are correctly defined. Supports multi-page templates.

---

## Settings

Access via **File > Settings** or the Settings page in the sidebar.

- **Default Paths** - Set default template and output directories.
- **Scoring Settings** - Threshold methods, fill thresholds, and fixed threshold values.
- **Alignment Settings** - DPI, alignment method, pass thresholds, and failure criteria.
- **Image/PDF Rendering** - DPI, image format, and quality settings for output PDFs.
- **PDF Annotation Settings** - Colors, thickness, font sizes, and box drawing options for annotated score sheets.

---

## Corrections Format

Corrections are stored in an append-only CSV file (`corrections.csv`) in each run folder. The first line records which results file the corrections apply to:

```
# applies to: /path/to/results.csv
timestamp,student_id,field,old_value,new_value
```

- Each edit appends a new row.
- A special `new_value` of `REVERT` undoes a previous correction.
- The effective correction for each student/field is the last non-REVERT entry.

---

## Answer Key Format

MarkShark supports a simple text-based answer key:

```
ver:A
A
B
C
D
A
ver:B
B
A
D
C
B
```

- Lines starting with `ver:` denote version headers.
- Each subsequent line is the correct answer for the next question.
- Single-version keys can omit the `ver:` header.

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+O | Open scans |
| Ctrl+, | Open Settings |
| Ctrl+Shift+R | Reset window size and centre |
| F1 | Open Help |

---

## Getting Help

- **GitHub:** [github.com/navarrew/markshark](https://github.com/navarrew/markshark)
- **Issues:** Report bugs or request features on the GitHub Issues page.

---

*This help file is bundled with your installed version of MarkShark. Click "Check for Latest" to fetch the most recent version from GitHub.*


HOW ALIGNMENTS WORK

1. Finding the markers (ArUco detection)
Each bubble sheet template has small square barcodes printed in known positions — these are ArUco markers, like tiny QR codes that each have a unique ID. When a student's scanned page comes in, the software looks for these markers in the image. Because they have distinct black-and-white patterns, the computer can find them even if the page is slightly rotated, shifted, or wrinkled.

2. Matching markers between template and scan
The software knows exactly where each marker should be on a perfect template. It matches up the markers it found in the student's scan with the expected positions by their ID numbers — marker #7 in the scan corresponds to marker #7 on the template. This gives it a set of "point pairs": where each marker is vs. where it should be.

3. Filtering out bad matches (RANSAC)
Sometimes a marker gets smudged, partially covered by a student's writing, or misidentified. RANSAC (Random Sample Consensus) is a method for ignoring these bad data points. It works by repeatedly picking a small random subset of marker pairs, computing a transformation from just those few points, and then checking how many of the other markers agree with that transformation. The transformation that gets the most "votes" wins. This way, even if a couple of markers are wrong or missing, the alignment still works because the good markers outvote the bad ones.

4. Warping the image
Once the best transformation is determined (a combination of rotation, scaling, and shifting), the entire scanned page is warped so that it lines up precisely with the template. Now every bubble on the scan sits exactly where the software expects it, and scoring can begin — the software just checks the darkness of each bubble at its known position.

In short: find the barcodes → match them to the template → use RANSAC to ignore bad ones → warp the whole page into alignment.
