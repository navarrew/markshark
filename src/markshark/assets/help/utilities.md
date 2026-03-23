# MarkShark Utilities

MarkShark includes several utilities to help you build answer keys, prepare your scans, verify your bubble sheet templates, and generate practice data for testing. You can find them all in the sidebar.

## Table of Contents
- [The Answer Key Utility](#the-answer-key-utility)
- [LMS Integration](#lms-integration)
- [PDF Tools](#pdf-tools)
- [The Bubblemap Visualizer](#the-bubblemap-visualizer)
- [The Mock Dataset Generator](#the-mock-dataset-generator)

---

# The Answer Key Utility

The Answer Key Utility is a visual tool for creating, editing, and exporting answer keys. You don't need this tool to make a key — a plain text file works fine — but this tool makes it easier to build one from scratch, verify it visually, and catch mistakes before you grade.

## Getting answers into the tool

There are two ways to start:

**Paste from another program.** Copy your answers from Word, Google Docs, Excel, or anywhere else. Paste them into the text box on the left side of the screen. MarkShark accepts answers separated by commas, spaces, tabs, semicolons, or line breaks — however you have them is fine. Click "Add to Key" and a short wizard will ask you to assign a version letter (A, B, etc.), an optional test code, and a default point value. The answers then appear in the table on the right.

**Import an existing key file.** Click "Import Key" to load a `.txt`, `.csv`, `.tsv`, or `.xlsx` file. The table fills in automatically. This is useful for reviewing or editing a key you've already created.

If you have an assessment selected and there's already a key file in its `input_files/` folder, MarkShark will notice and show a yellow bar at the top offering to load it for you.

## Editing in the table

The table works like a simple spreadsheet. Each column pair represents one version — the left column shows the answer and the right column shows the point value.

- **Double-click** any cell to edit it. Press Enter to confirm and move down, or Tab to move right.
- **Answers are colour-coded** — blue for single answers, amber for multi-answer operators like `A^B`, green for freebies (`*`), and grey for discarded (blank) questions.
- **Right-click** for more options: insert or delete a row, shift one version's answers down (handy if you need to add a question in the middle of one version without affecting the others), or edit a version's header to change its test code or default points.
- **Point values** can be overridden per question. If you change a version's default points, questions that were still at the old default update automatically — questions you've already customized are left alone.

A summary panel on the left updates in real time to show how many versions you have, how many questions, and the total points per version.

## Exporting your key

When you're happy with the key, you have two options:

- **Save to Assessment** writes `key.txt` into the selected assessment's `input_files/` folder, ready for scoring.
- **Save As** lets you choose a custom filename and location. You can export as either `.txt` (text format) or `.xlsx` (Excel workbook).

Before saving, MarkShark checks for empty answer cells and warns you. Empty cells are treated as discarded questions, so this is just a heads-up in case you left one blank by accident.

## Answer syntax reference

The right side of the screen includes a quick-reference panel showing all the answer operators MarkShark supports (`A^B` for either/or, `A&B` for must-select-both, `*` for freebie, etc.). For the full list with examples, see [Answer Key Formats](key_formats.md#advanced-scoring-options).

---

# LMS Integration

The LMS Integration page bridges your Learning Management System gradebook with MarkShark. It has two tabs: **Import & Map Columns** lets you load a gradebook export, map its columns to MarkShark properties, save the mapping as a reusable filter, and export a roster. **Write Scores Back** lets you insert MarkShark scores into a copy of your LMS gradebook file so you can upload it back to your LMS.

## Import & Map Columns

### Step 1 — Select a Gradebook File

Click the file selector to load a gradebook exported from your LMS. Supported formats are CSV, TSV, TAB, XLSX, and XLS. A preview table shows the first several rows so you can verify the data before mapping.

### Step 2 — Adjust Parse Settings

- **Delimiter** — choose comma, tab, semicolon, or pipe. This is ignored for Excel files.
- **Skip rows** — some LMS exports include extra header rows (instructions, dates, etc.) before the actual column names. Set this to skip past them.

After changing either setting, click **Reload Preview** or the preview will update automatically.

### Step 3 — Map Columns

Assign which columns in your gradebook correspond to MarkShark properties:

- **Student ID** — the unique identifier that matches MarkShark's StudentID field.
- **Last Name** — student's surname.
- **First Name** — student's given name.
- **Combined Name** — for LMS exports that put both names in a single column (e.g. Canvas "Sortable Name" in "Last, First" format). MarkShark splits on the comma to extract last and first names. Leave this as "(none)" if your export has separate name columns.

MarkShark auto-detects common column names. For example, a column called "SIS User ID" is automatically mapped to Student ID, and "Sortable Name" is mapped to Combined Name.

### Step 4 — Save as Filter

Click **Save As…** to save this column mapping as a named filter. Next time you export from the same LMS, simply load the filter instead of re-mapping. See [Saved Filters](#saved-filters) below.

### Step 5 — Export Roster

Click **Export as MarkShark Roster** to generate a `roster.csv` file with three columns: StudentID, LastName, and FirstName. This roster works with all MarkShark features that accept a roster file (Grader, Report, Reorder to Roster, etc.).

A Student ID mapping is required before exporting.

## Write Scores Back

After scoring and reviewing corrections, use this tab to write final grades back into a copy of your LMS gradebook.

### Inputs

1. **LMS gradebook** — select the original gradebook file exported from your LMS.
2. **Exam report** — select the `exam_report.xlsx` generated by MarkShark. This is the report that includes any corrections made on the Review & Correct page, so you get final grades.

### Mapping Options

- **Use saved filter** — select a filter you created on the Import & Map tab. Click **Apply Filter to SID Column** to apply it. This sets the delimiter, skip rows, and Student ID column automatically.
- **LMS Student ID column** — which column in the LMS file contains the student identifier. This is populated automatically when you load a file or apply a filter.
- **Value to write** — choose **Score (raw)** for the number of correct answers, or **Percent** for the percentage score.
- **Target column** — choose an existing column in the gradebook to overwrite, or select **"➕ Add a new column…"** to create a new one (e.g. "Exam 1"). When adding a new column, MarkShark will prompt you for the column name.
- **Absent students** — choose **Leave blank** to skip students who have no score, or **Enter zero (0)** to write a zero in their cell. Cells that already have content are never overwritten.

### Orphan Warning

If MarkShark detects orphan students in the exam report (students whose IDs didn't match the roster), it will warn you before writing. Orphan scores can't match any LMS entry and will be skipped. Consider correcting orphan IDs on the Review & Correct page first.

### Output

Choose a save path for the updated gradebook. The output format is determined by the file extension — `.csv` preserves the original delimiter, `.xlsx` creates a new Excel file. MarkShark matches students by their Student ID and fills in the score values. After writing, a status message shows how many students were matched (e.g. "Done — 28/30 students matched").

Scores are read from the **Class Scores** sheet of the exam report, which reflects all manual corrections.

## Saved Filters

Filters are stored in `~/.markshark/lms_filters.json` and persist across sessions. Each filter saves:

- Column assignments (Student ID, Last Name, First Name, Combined Name)
- Delimiter setting
- Skip rows count

Filters are shared between both tabs — a filter created on the Import & Map tab can be applied on the Write Scores tab.

You can **Load**, **Save As…** (create or overwrite), or **Delete** filters from the Saved Filters section on the Import & Map tab.

## Common LMS Export Formats

| LMS | Typical Format | Student ID Column | Notes |
|-----|---------------|-------------------|-------|
| **Canvas** | CSV | SIS User ID | "Sortable Name" column is "Last, First" — use Combined Name mapping |
| **Blackboard** | CSV or XLS | Student ID | |
| **Moodle** | CSV | ID number | |
| **Brightspace (D2L)** | CSV | OrgDefinedId | |
| **Google Classroom** | CSV | Student ID | |

## LMS Tips

- **Always preview** your file after loading to verify the columns look correct before mapping.
- **Save a filter per LMS** — if you use Canvas, save a "Canvas" filter. If you switch platforms, create a new filter rather than overwriting the old one.
- **Roster reuse** — exported rosters work with all MarkShark features that accept a roster file (Grader, Report, Reorder to Roster, etc.).
- **Write scores last** — generate your report and review it before writing scores back. The Write Scores tab reads from the exam report, which includes corrections. If you discover more errors after writing, just regenerate the report and write again.
- **Don't overwrite your original** — save the output to a new file so you always have the original LMS export as a backup.

---

# PDF Tools

The PDF Tools page has four tabs for common scan preparation tasks. You shouldn't need these often, but they're here when you do.

## Convert Images to PDF

Turns a folder of scanned images (JPG, PNG, TIFF, or BMP) into a single multi-page PDF. Select the folder, choose whether to sort pages by filename or by date modified, pick an output path, and click Convert. This is useful if your scanner saves each page as a separate image file instead of combining them into one PDF.

## Combine PDFs

Merges multiple PDF files into a single PDF. Add files with the "Add PDF" button, drag them into the order you want (or use the Move Up / Move Down buttons), then click Merge. Useful when you scanned different batches of answer sheets separately and need to combine them into one file for grading.

## Sort or Reorder Pages

Reorders the pages of a scored PDF based on your results CSV. Load the scored PDF and the `results.csv` from scoring, choose whether to sort by last name or student ID, and click Reorder. The output is a new PDF with student sheets in alphabetical or ID order — handy for returning graded sheets to students or filing them in order.

## Interdigitate Scans

Interleaves pages from two separate PDFs to reconstruct a double-sided scan. If your document feeder scans front sides and back sides as two separate stacks, this tool reassembles them into the correct front-back-front-back order. The "Reverse back pages" checkbox (on by default) handles the common situation where the back-side stack comes out in reverse order after flipping. If your front PDF has 20 pages and your back PDF has 20 pages, the output will be a 40-page PDF with each student's front and back pages together.

---

# The Bubblemap Visualizer

Every bubble sheet template in MarkShark has two parts: a PDF of the printed sheet and a bubblemap file (a YAML text file) that tells the software exactly where each bubble is on the page. The Bubblemap Visualizer overlays the bubblemap's grid onto a PDF so you can visually verify that the bubble positions line up correctly.

## When to use it

- **Checking a new template.** If you've installed a new bubble sheet template or created a custom bubblemap, overlay it onto the blank template PDF to confirm the circles land exactly on the printed bubbles.
- **Troubleshooting alignment.** If scoring is misreading certain bubbles, the visualizer can show you whether the bubblemap definition is off or whether the problem is elsewhere (scan quality, rotation, etc.).
- **Checking aligned scans.** You can overlay the bubblemap onto your aligned scans (not just the blank template) to see how the alignment positioned each student's sheet relative to the expected bubble locations.

## How it works

Select a bubblemap from the dropdown (it lists all your installed templates) or browse for a custom YAML file. Then select a PDF — this can be the blank template, a set of aligned scans, or even a scored PDF. Adjust the DPI slider if you want higher or lower resolution, then click "Visualize Bubblemap."

MarkShark draws coloured circles at every bubble position defined in the bubblemap. You can page through multi-page templates, zoom between fit-to-window and full size, and save any page as a PNG or JPEG image if you want to keep a copy for reference or share it with a colleague.

---

# The Mock Dataset Generator

The Mock Dataset Generator creates a complete synthetic dataset — fake student scans, an answer key, a roster, and a response table — from any installed template. It's useful for testing your workflow end-to-end without needing real student data, for training new users, or for trying out a new template before printing and using it in a real exam.

## What it generates

- **Scanned PDFs** — simulated student answer sheets with realistic bubble fills, slight rotation, and varying pencil darkness
- **Answer key** — a text file with correct answers for each version
- **Roster CSV** — a class list with student IDs, names, and version assignments
- **Responses CSV** — a table showing what each simulated student answered

## Configurable parameters

The left side of the screen controls the dataset:

- **Students** — how many fake students to generate (1–500)
- **Versions** — how many exam versions (1–10, if the template supports a version zone)
- **Absent students** — students who appear in the roster but have no submitted scan
- **Random seed** — use the same seed to reproduce identical results (useful for tutorials)

Below those are error simulation settings that make the data realistic:

- **Blank answer rate** — percentage of wrong answers left empty
- **Multi-fill rate** — percentage of wrong answers where the student filled multiple bubbles
- **ID mis-entries** — students with corrupted IDs (typos, missing digits)
- **Missing version** — students who forgot to fill in their version bubble

The right side controls scan quality (DPI, minimum bubble darkness, random rotation) and answer key options (AND/OR questions, weighted questions, default points).

## Using the output

Pick an output folder, click "Generate Mock Dataset," and MarkShark will create all the files. You can then run the full grading workflow — align, score, review, report — on this synthetic data to see exactly how MarkShark handles various situations (blanks, multi-fills, missing IDs, absent students) without any risk to real grades.
