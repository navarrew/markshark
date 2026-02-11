# Getting Started
MarkShark is an optical mark recognition (OMR) system for bubble sheet grading.
---
## Typical Workflow
1. **Scan** your bubble sheets to a single PDF (use a document scanner or scanning app).
2. **Open MarkShark** and go to the **Grader** page.
3. **Select your project folder** (or create a new one).
4. **Load your scans**, answer key, and optionally a class roster.
5. Click **Align & Score** — MarkShark will align and score your sheets.
6. **Review & Correct** flagged items on the Review page.
7. **Generate a report** with statistics, item analysis, and per-student details.
---
## Project Structure
MarkShark organises files into projects. A project is typically a single test (e.g. "BIO101 2025 Midterm 1"). You set a working directory for your class and create projects inside it.
```
BIO101/                       # working directory
  midterm_1/                  # project folder
    input_files/              # your uploaded files
      raw_scans.pdf
      answer_key.txt
      roster.csv
    score_data/               # grading outputs
      results.csv
      corrections.csv
    exam_report.xlsx          # generated report
    scored_scans.pdf          # annotated PDF
    aligned_scans.pdf         # aligned PDF
    logs/                     # processing logs
  midterm_2/                  # another project
    ...
```
### input_files/
Where your original uploads are stored: scanned PDFs, answer keys, and rosters.
### score_data/
Where grading outputs live: the raw `results.csv` and any `corrections.csv` from the Review page.
### Top-level outputs
The `exam_report.xlsx`, `scored_scans.pdf`, and `aligned_scans.pdf` are placed at the project root for easy access.
---
## Pages Overview
### Grader
The main grading page. Load scans, select a template, provide an answer key and optional roster, then click Align & Score. After scoring, switch to the Generate Report tab to create an Excel report.
### Review & Correct
View scored results in a spreadsheet alongside the scanned PDF. Click on a student row to see their sheet. Double-click answer cells to correct them. Corrections are saved automatically.
### Template Manager
Browse, preview, favourite, and reorder your installed bubble sheet templates. Each template includes a PDF master and a bubblemap YAML that defines bubble positions.
### Project Manager
Manage project folders, view project contents, and open projects directly in the Grader.
---
## Standalone Tools
### Align Only
Run the alignment step independently. Useful for troubleshooting alignment issues or preparing scans for external processing.
### Score Only
Run scoring on pre-aligned PDFs without re-running alignment. Useful for re-scoring with different thresholds.
### Report Only
Generate a report from existing results and corrections without re-running the full pipeline.
---
## Utilities
### Mock Data Utility
Generate synthetic student datasets for testing. Creates fake scans, answer keys, response CSVs, and rosters from any installed template. Useful for testing templates before real grading.
### Map Viewer
Overlay bubblemap grid onto a template PDF to visually verify that bubble positions are correctly defined. Supports multi-page templates.
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
