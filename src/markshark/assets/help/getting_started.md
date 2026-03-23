# Getting Started
**Note:** *If you're new to MarkShark we have a tutorial PDF and a practice dataset for you to download and experiment with on the welcome page.*

## Table of Contents
- [Step 1 - Find a bubble sheet](#step-1---find-a-bubble-sheet-that-meets-your-needs)
- [Step 2 - Set up your folders](#step-2---set-up-your-markshark-folders-on-your-computer)
- [Step 3 - Workflow after the test](#step-3---workflow-after-the-test)
- [Scanning tips](#scanning-tips)
- [Your answer key](#your-answer-key)
- [Your class roster](#your-class-roster-optional)
- [MarkShark folders and files](#more-about-the-markshark-folders-and-files)
- [An overview of MarkShark pages](#an-overview-of-markshark-pages)
- [Getting help](#getting-help)

# Step 1 - Find a bubble sheet that meets your needs.
MarkShark comes with many different bubble sheets templates you can download, print, and use right away.  MarkShark bubble sheet templates comprise two files - a pdf and an associated text file (a **bubblemap**) that tells the software what each of the bubbles represents.
On the **Welcome page** you can click 'Browse Templates' and see the various bubble sheets that can be used right away with MarkShark.  In the **Template Manager** you can get a closer look at each of the templates, select some favorites, and archive the ones you don't want to use to get them out of the way.  Click on the 'Download PDF' button and you can print and use these bubble sheets right away!
***Can I continue to use a non MarkShark bubble sheet I've already been using?***  
*Uhhhh...yes and no.*  MarkShark can't decode a bubble sheet just by looking at it.  Every bubble sheet needs a corresponding **bubblemap** file that tells the software where the bubbles are on the page and what they represent.
It's not *that* hard to make a bubblemap for a pre-existing bubble sheet (its just a text file and we give you instructions) but it does take some effort up front.  Once you make a bubblemap for a particular bubble sheet you can reuse it forever.  You'll have to decide for yourself whether the up-front effort (an hour) to make an accurate bubblemap for your old bubblesheet is worth it.
# Step 2 - Set up your MarkShark folder(s) on your computer.
You should put a MarkShark folder inside the folder you might already use for your course files (for your lectures, syllabi, notes, etc.).  You don't have to name the folder MarkShark...you can call it 'tests' or 'test scores' or 'section 101 scores'.  Doesn't matter.  MarkShark will remember where it is.  You can have several MarkShark folders even within the same course folder (one for each section of a large class, for example).
For each new assessment (quiz, test, exam) MarkShark will create a new 'assessment' subfolder where it will automatically store the scans, keys, rosters, and output reports/scores associated with that particular assessment.
In the example below, the teacher of Biology 101 has put two MarkShark folders (red) inside their course's main folder, one for each of the sections they teach.  They did this because the two sections enroll different students and it was important to keep their marks separate (you could decide to merge course sections if you so choose).  Inside each of the MarkShark folders are folders for each of the tests held during the course.  Looking inside the midterm 1 folder you see the 'input_files' folder holds a pdf of the scans, the midterm 1 answer key, and the class roster.  The scored_scans.pdf and final report output are kept in the main assessment folder.  Other files MarkShark needs are kept in the logs and score_data folders.
![MarkShark folder structure](img/folders.png)
# Step 3 - workflow after the test
So you're back in your office after the test. What's next?
1. **Scan** your student bubble sheets as a single PDF. [*See scanning tips below*](#scanning-tips).
2. **Open MarkShark** and go to the **Grader** page.
3. **Create a new assessment within your course's MarkShark folder**
4. **Upload your scans, answer key, and optionally a class roster.** [*See how to create your key and class roster*](key_formats.md)
5. Click **Align & Score** — MarkShark will align and score your sheets.
6. **Review & Correct** flagged items on the Review & Correct page.
7. If you make corrections, click **Apply Corrections & Re-annotate** to update the scored PDF and CSV.
8. **Generate a report** with statistics, item analysis, and per-student details.
![Annotated PDF example](img/processmap.png)
## Scanning tips
The best way to avoid issues is to feed MarkShark good scans of your student bubblesheets.  With good scans you will encounter few, if any, issues.
- **Use a scanner with a document feeder.** Many multi-function laser printers and photocopiers can work to get you scans that are 
- **Scan using grayscale, at a resolution of 150 dpi.**  There's not much benefit having a resolution greater than 150 dots per inch and a 300 dpi image is 4x larger in terms of file size.  It's unlikely that your scans will perform significantly better with a higher resolution, but don't go much lower than 150 dpi either.
- **Do not use a phone camera.**  Photos taken with a phone warp the image unless you are looking straight down upon the scan and the paper is flat, not curled.  Lighting across the picture is often uneven and shadows from your arms or the camera can make certain areas of the scan much darker than others.  Any time you save using your phone will cost you later in troubleshooting how to de-warp the poor-quality scan.
- **Use white paper for your bubble sheets.** MarkShark must distinguish between gray pixels that are background vs. gray pixels that were caused by a pencil mark.  If your background paper is colored it will often appear as gray when converted to black and white.  You want to maximize the difference between the gray-level on the paper and the pencil marks written by the student.
## Your answer key
You'll need a file that lists the correct answer for each question. The easiest way to make one is to download our sample key from the Welcome page and replace the answers with your own. You can also paste answers from Word, Excel, or Google Docs into the Answer Key Utility, or create a simple text file or spreadsheet from scratch. MarkShark accepts `.txt`, `.csv`, and `.xlsx` key files and handles single-version or multi-version exams.

For the full details on key formats and scoring options, see [Answer Keys and Rosters](key_formats.md).

## Your class roster (optional)
A class roster is a CSV file with student IDs and names. If you provide one, MarkShark will match scanned sheets to students by ID, display names in the results, and flag anyone who was absent. Most learning management systems can export a class list as CSV that works with MarkShark. A roster is optional — without one, MarkShark still scores every sheet but can only identify students by what they bubbled in.

For roster column details, see [Answer Keys and Rosters](key_formats.md#your-class-roster-optional).

---
# More about the MarkShark folders and files
## MarkShark File Locations
MarkShark will automatically make copies of your scans, key, and roster and put them in the folder 'input_files' in your assessment folder.  It also saves the aligned scans to the input_files folder after running an alignment.  Throughout the grading and reporting processes, MarkShark saves a variety of files to different places in the assessment folder.  
MarkShark does not modify your original files, but instead makes copies of them to the input_files directory and works with those copies.  This is to protect your original scan and answer key files, making it easy to retry if something goes wrong along the way.
| File | Location | Purpose |
|------|----------|---------|
| `scored_scans.pdf` | assessment root | Annotated PDF with scoring marks |
| `exam_report.xlsx` | assessment root | Generated Excel report |
| `raw_scans.pdf` | `input_files/` | Original scanned bubble sheets |
| `aligned_scans.pdf` | `input_files/` | Aligned PDF (no scoring marks) |
| `answer_key.txt` | `input_files/` | Answer key file |
| `roster.csv` | `input_files/` | Class roster |
| `results.csv` | `score_data/` | Scoring results |
| `results_original.csv` | `score_data/` | Backup of results before re-annotation |
| `results_params.json` | `score_data/` | Scoring parameters (for re-annotation) |
| `corrections.csv` | `score_data/` | Manual corrections log |
| `log_DATE-TIME.txt` | `logs/` | Logs of alignment and scoring data |
---
## An Overview of MarkShark Pages

The pages below are listed in the same order as the sidebar. Most teachers only use the first few regularly — the rest are there when you need them.

### Welcome
Your starting point. Browse bubble sheet templates, download sample data, and access the tutorial. If you're new to MarkShark, start here.

### Grader
The main grading page. Load scans, select a template, provide an answer key and optional roster, then click **Align & Score**. After scoring, switch to the **Generate Report** tab to create an Excel report. See [Scoring](scoring.md) and [Report](report.md) for details.

### Review & Correct
View scored results in a spreadsheet alongside the annotated PDF. Click on a student row to see their scanned sheet. Double-click answer cells to correct misread answers or student IDs. When corrections are ready, click **Apply Corrections & Re-annotate** to update both the CSV and the annotated PDF. See [Corrections](corrections.md) for details.

---

### Align Only
Run the alignment step by itself, without scoring. Useful for troubleshooting alignment issues or preparing scans for external processing.

### Score Only
Run scoring on pre-aligned PDFs without re-running alignment. Useful for re-scoring with a different answer key or different threshold settings.

### Report Only
Generate a report from an existing `results.csv` without re-running alignment or scoring. Handy if you just want to regenerate the Excel report with different options.

---

### Course Manager
Organize your courses and assessments. Browse course folders, view assessment contents, create new assessments, and open them directly in the Grader.

### Template Manager
Browse, preview, favourite, and reorder your installed bubble sheet templates. Each template includes a printable PDF and a bubblemap file that tells MarkShark where the bubbles are.

---

### Answer Key Utility
A visual tool for building, editing, and exporting answer keys. Paste answers from Word or Excel, or import an existing key file. See [Utilities](utilities.md#the-answer-key-utility) for details.

### LMS Integration
Import and export gradebook files for learning management systems (Canvas, Brightspace, etc.). Create reusable column-mapping filters and write MarkShark scores back into your LMS gradebook. See [Utilities](utilities.md#lms-integration) for details.

### PDF Tools
Convert images to PDF, combine multiple PDFs, reorder pages, or interleave front-and-back scans. See [Utilities](utilities.md#pdf-tools) for details.

### Mock Data Utility
Generate a complete synthetic dataset (fake scans, answer key, roster) from any installed template. Useful for testing your workflow or following the tutorial without needing real student data.

### Bubblemap Utility
Overlay a bubblemap grid onto a PDF to visually verify that bubble positions are defined correctly. Useful for checking new templates or troubleshooting alignment issues. See [Utilities](utilities.md#the-bubblemap-visualizer) for details.

---

### Help
The help pages you're reading right now.

### Settings
Configure default paths, scoring thresholds, alignment parameters, and display options.

