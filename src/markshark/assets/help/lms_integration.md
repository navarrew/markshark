# LMS Integration
Use the LMS Integration page to bridge your Learning Management System gradebook exports with MarkShark. Import a gradebook, map its columns, save the mapping as a reusable filter, and optionally write scores back.
---
## Import & Map Columns
### Step 1: Select a Gradebook File
Upload a CSV, TSV, or XLSX file exported from your LMS (Canvas, Blackboard, Moodle, etc.). The file preview shows the first several rows so you can verify the data.
### Step 2: Adjust Parse Settings
- **Delimiter** — Choose comma, tab, semicolon, or pipe. Ignored for XLSX files.
- **Skip rows** — Some LMS exports include extra header rows before the actual column names. Set this to skip them.
### Step 3: Map Columns
Assign which columns in your gradebook correspond to:
- **Student ID** — The unique identifier that matches MarkShark's StudentID field
- **Last Name** — Student's surname
- **First Name** — Student's given name

MarkShark auto-detects common column names (e.g. "SIS User ID", "Last Name", "First Name").
### Step 4: Save as Filter
Click **Save As...** to save this column mapping as a named filter. Next time you export from the same LMS, simply load the filter instead of re-mapping columns.
### Step 5: Export Roster
Click **Export as MarkShark Roster** to generate a `roster.csv` file that works directly with MarkShark's grading workflow.
---
## Saved Filters
Filters are stored in `~/.markshark/lms_filters.json` and persist across sessions. Each filter saves:
- Column assignments (Student ID, Last Name, First Name)
- Delimiter setting
- Skip rows count

You can load, overwrite, or delete filters from the Saved Filters section.
---
## Write Scores Back
After scoring and generating a report with MarkShark, you can write final scores back into your LMS gradebook format:
1. Select the **original LMS gradebook** file
2. Select the **exam_report.xlsx** from a scored project — this is the report that includes any corrections made on the Review panel, so you get final grades
3. Choose a saved filter (or manually set the Student ID column)
4. Select whether to write the **raw score** or **percent**
5. Enter a **target column name** (e.g. "Exam 1") — if the column exists it will be updated, otherwise a new column is added
6. Choose what to do for **absent students** — leave blank or enter zero
7. Choose an output path and click **Write Scores**

The output file matches students by their Student ID and fills in the score values. Scores are read from the "Class Scores" sheet of the exam report, which reflects all manual corrections.
---
## Common LMS Export Formats
| LMS | Typical Format | Student ID Column |
|-----|---------------|-------------------|
| **Canvas** | CSV | SIS User ID |
| **Blackboard** | CSV or XLS | Student ID |
| **Moodle** | CSV | ID number |
| **Brightspace (D2L)** | CSV | OrgDefinedId |
| **Google Classroom** | CSV | Student ID |
---
## Tips
- **Always preview** your file after loading to verify the columns look correct before mapping.
- **Save a filter per LMS** — if you use Canvas, save a "Canvas" filter. If you switch LMS platforms, create a new filter.
- **Roster reuse** — exported rosters work with all MarkShark features that accept a roster file (Grader, Reorder to Roster, etc.).
