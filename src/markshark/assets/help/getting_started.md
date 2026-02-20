# Getting Started
MarkShark is an optical mark recognition (OMR) system for bubble sheet grading that is designed to be teacher friendly.
***It's completely free.  Really.***  
- No paywalls or microtransactions.
- No licensing requirements.
- No arguing with your school about the costs of scanning.
- You print your own bubble sheets, you don't have to order them.
- You can use a scanner you probably already have access to.
**MarkShark makes it easy for you.**
- MarkShark works on both Mac and PC computers.
- You control the data and can go from scans to scores in minutes.
- We provide dozens of bubblesheet templates for you to choose from.
  - *You can your own custom bubble sheets using our companion platform 'Bubblefish'*
- Make a mistake with your answer key?  No problem!  You can rescore in seconds!
- A student fill in their ID incorrectly?  No problem!  You can fix it with a click!
- You scanned one page upside-down?  No problem!  Our PDF tools can fix it fast!
- You want to copy student scores directly into the spreadsheet your course software uses?  No problem!  It's easy to integrate MarkShark with any LMS!
- MarkShark works for courses as small as 10 students to classes bigger than 1000!
- MarkShark accepts a variety of answer key formats.
- MarkShark generates a report that is easy to read and understand.
- MarkShark produces annotated student scans make it easy to see and understand why a student got their score.
# Step 1 - Find a bubble sheet (or two) that meet your needs.

# Step 2 - Set up your MarkShark folder on your computer.


[see Getting Started](getting_started.md)
[see Troubleshooting](scoring.md#troubleshooting)
[Answer Key Formats](key_formats.md)
---
# Typical workflow after the test
1. **Scan** your student bubble sheets to a single PDF (use a document scanner or scanning app).
2. **Open MarkShark** and go to the **Grader** page.
3. **Create a new assessment within your MarkShark course folder**
4. **Load your scans, answer key, and optionally a class roster.**
5. Click **Align & Score** — MarkShark will align and score your sheets.
6. **Review & Correct** flagged items on the Review & Correct page.
7. If you make corrections, click **Apply Corrections & Re-annotate** to update the scored PDF and CSV.
8. **Generate a report** with statistics, item analysis, and per-student details.
![Annotated PDF example](img/processmap.png)
---
# Where to keep your MarkShark files
MarkShark organises files into assessments. An assessment is typically a single test (e.g. "BIO101 2025 Midterm 1"). You set a course folder for your class and create assessments inside it.
```
BIO101 Section/               # the folder you already use for all your course content
  MarkShark/                  # A folder inside your course folder specific to MarkShark
    midterm_1/                # assessment folder for midterm #1
      input_files/            # inputs + aligned scans
        raw_scans.pdf
        aligned_scans.pdf
        answer_key.txt
        roster.csv
      score_data/               # grading outputs
        results.csv
        results_original.csv
        results_params.json
        corrections.csv
      scored_scans.pdf          # annotated PDF (project root)
      exam_report.xlsx          # generated report
      logs/                     # processing logs
    midterm_2/                  # another assessment folder...for midterm #2
      ...
```
### input_files/
Where your original uploads are stored: scanned PDFs, answer keys, and rosters. The aligned PDF (`aligned_scans.pdf`) is also saved here after alignment.
### score_data/
Where grading outputs live: the scored `results.csv`, scoring parameters (`results_params.json`), and any `corrections.csv` from the Review page. If you re-annotate, the original results are archived as `results_original.csv`.
### Top-level outputs
The `scored_scans.pdf` (annotated PDF with scores) and `exam_report.xlsx` are placed at the assessment root for easy access.
---
## An Overview of MarkShark Pages
### Welcome
XXX.
### Grader
The main grading page. Load scans, select a template, provide an answer key and optional roster, then click **Align & Score**. After scoring, switch to the **Generate Report** tab to create an Excel report.
### Review & Correct
View scored results in a spreadsheet alongside the annotated PDF. Click on a student row to see their scanned sheet. Double-click answer cells to correct misread answers or student IDs. Corrections are saved automatically.

When corrections are ready, click **Apply Corrections & Re-annotate** to re-run scoring with your corrections applied. This updates both the CSV scores and the annotated PDF — corrected answers are marked with teal diamonds and a "Corrections applied" stamp. Use **Clear Corrections** to start fresh if needed.
---
## Management
### Course Manager
Manage your courses and assessments. Browse course folders, view assessment contents, create new assessments, and open them directly in the Grader.
### Template Manager
Browse, preview, favourite, and reorder your installed bubble sheet templates. Each template includes a PDF master and a bubblemap YAML that defines bubble positions.
### LMS Integration
Import and export gradebook files for learning management systems (Canvas, Brightspace, etc.).
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
### Answer Key Utility
Create, import, edit, and export answer keys. Supports importing from text files, CSVs, and Excel spreadsheets. Build keys from scratch with per-question answer selection and multi-answer support.
### PDF Tools
Split, merge, and manipulate PDF files. Useful for preparing scans or combining output documents.
### Mock Data Utility
Generate synthetic student datasets for testing. Creates fake scans, answer keys, response CSVs, and rosters from any installed template. Useful for testing templates before real grading.
### Bubblemap Utility
Overlay the bubblemap grid onto a template PDF to visually verify that bubble positions are correctly defined. Supports multi-page templates and shows the output zone location.

---
## Getting Help
- **GitHub:** [github.com/navarrew/markshark](https://github.com/navarrew/markshark)
- **Issues:** Report bugs or request features on the GitHub Issues page.
