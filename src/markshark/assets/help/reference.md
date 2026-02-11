# Reference
Quick reference for file formats, output columns, settings, and other details.
---
## Results CSV Columns
The `results.csv` file in `score_data/` contains one row per student:
| Column | Description |
|--------|-------------|
| Page | Page number in the scanned PDF |
| Version | Detected test version (A, B, C, etc.) |
| LastName | Student's last name (from bubbled name or roster) |
| FirstName | Student's first name |
| StudentID | Student ID (from bubbled ID field) |
| Correct | Number of correct answers |
| Incorrect | Number of incorrect answers |
| Blank | Number of blank (unanswered) questions |
| Multi | Number of multi-fill (ambiguous) questions |
| Score | Percentage score |
| Points | Total points earned |
| MaxPoints | Maximum possible points |
| Flags | Comma-separated list of flag codes |
| Q1, Q2, ... | Student's detected answer for each question |
---
## Excel Report Contents
The `exam_report.xlsx` contains multiple sheets:
- **Summary** — Class statistics: mean, median, standard deviation, score distribution
- **Student Scores** — Per-student results with names, IDs, scores, and flags
- **Item Analysis** — Per-question statistics: difficulty index, discrimination, answer distribution
- **Flagged Items** — Students/questions needing attention
- **Absent Students** — Students in the roster who had no scanned sheet (if roster provided)
---
## Settings Reference
### Default Paths
- **Template directory** — Where MarkShark looks for installed templates
- **Output directory** — Default location for new projects
### Scoring Settings
- **Threshold method** — How the fill threshold is determined
- **Min fill threshold** — Minimum darkness for a bubble to count as filled
- **Fixed threshold** — Gray-level binarisation threshold
### Alignment Settings
- **DPI** — Resolution for rendering PDF pages
- **Alignment method** — auto, fast, slow, or aruco
- **Pass thresholds** — Quality criteria for alignment success
### Image/PDF Rendering
- **DPI** — Output resolution for annotated PDFs
- **Image format** — Internal rendering format
- **Quality** — JPEG quality (if applicable)
### PDF Annotation Settings
- **Colours** — Colours for correct, incorrect, blank, multi markers
- **Thickness** — Line thickness for bubble outlines
- **Font sizes** — Text size for fill percentage labels
- **Box drawing** — Whether to draw bounding boxes around zones
---
## Common File Locations
| File | Location | Purpose |
|------|----------|---------|
| `raw_scans.pdf` | `input_files/` | Original scanned bubble sheets |
| `answer_key.txt` | `input_files/` | Answer key file |
| `roster.csv` | `input_files/` | Class roster |
| `results.csv` | `score_data/` | Raw scoring results |
| `corrections.csv` | `score_data/` | Manual corrections log |
| `exam_report.xlsx` | project root | Generated Excel report |
| `scored_scans.pdf` | project root | Annotated PDF with scoring marks |
| `aligned_scans.pdf` | project root | Aligned PDF (no scoring marks) |
---
## Command Line Usage
MarkShark can be run from the command line for batch processing:
```
markshark align <scans.pdf> --template <template.pdf> --out-pdf aligned.pdf
markshark score <aligned.pdf> --bublmap <bubblemap.yaml> --out-csv results.csv
markshark report <results.csv> --out-xlsx report.xlsx
```
Run `markshark --help` for the full list of options.
---
## Supported Scan Formats
- **PDF** — Preferred format. Multi-page PDFs with one student per page (or per page-pair for multi-page templates).
- Scans should be **grayscale or colour** (not black-and-white / 1-bit).
- Recommended scanning resolution: **200-300 DPI** at the scanner. MarkShark re-renders at its own DPI internally.
---
## Tips for Best Results
### Scanning
- Use a document feeder scanner for speed.
- Scan all sheets for one test into a single PDF.
- Keep scanner settings consistent across the batch.
- Avoid scanning in pure black-and-white mode — grayscale preserves bubble fill information.
### Answer Keys
- Double-check your answer key before grading. A wrong key affects every student.
- Use the text format for simple tests, CSV/Excel for multi-version.
- Take advantage of OR mode (`A^B`) when two answers should be accepted.
- Use freebie (`*`) to give everyone credit for a bad question.
### Review
- Always review flagged items before generating the final report.
- Use the scored PDF to visually check alignment quality.
- Look for patterns in flags — many blanks may indicate a threshold issue, not student errors.
