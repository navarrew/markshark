# Answer Keys and Rosters

MarkShark is designed to work with however you prefer to create your answer key. You don't need to learn a special format — just get your answers into a file and MarkShark will figure out the rest.

## Table of Contents
- [The Easy Way](#the-easy-way)
- [Supported File Types](#supported-file-types)
- [Single-Version Keys](#single-version-keys)
- [Multi-Version Keys](#multi-version-keys)
- [Version Headers Explained](#version-headers-explained)
- [Advanced Scoring Options](#advanced-scoring-options)
- [Scoring Modes Summary](#scoring-modes-summary)
- [Your Class Roster](#your-class-roster-optional)

---

## The Easy Way

Most teachers find one of these approaches quickest:

**Start from our sample key.** On the Welcome page, click "Download Sample Key" to get a pre-formatted text file with two versions already set up. Open it in any text editor, replace our answers with yours, and save. The sample file includes helpful comments (lines starting with `##`) that explain the format as you go — MarkShark ignores these lines, so you can leave them in or delete them.

**Paste into the Answer Key Utility.** If you already have your answers in a Word document, Google Doc, or spreadsheet, copy the column or row of answers and paste it into the Answer Key Utility (available from the Welcome page or the sidebar). MarkShark will parse answers separated by commas, tabs, spaces, or newlines. A short wizard lets you assign a version letter and point value, then you can export the finished key as a text file or Excel workbook.

**Use a spreadsheet.** If you prefer working in Excel or Google Sheets, create a simple table with one column per version and one row per question. Save as `.csv` or `.xlsx` and MarkShark will read it directly. See [CSV / TSV Format](#csv--tsv-format) and [Excel Format](#excel-format-xlsx) below.

---

## Supported File Types

| Format | Extension | Best for |
|--------|-----------|----------|
| Text | `.txt` | Quick single-version keys, easy to edit in any text editor |
| CSV | `.csv` | Multi-version keys where you want answers side-by-side |
| TSV | `.tsv` | Same as CSV but tab-separated (copy-paste from Excel) |
| Excel | `.xlsx` | Multi-version keys if you prefer working in a spreadsheet |

MarkShark auto-detects the format from the file extension, so just make sure your file is saved with the right one.

---

## Single-Version Keys

If your test has only one version, your key can be as simple as a list of correct answers. No header needed.

**One answer per line:**
```
A
B
C
D
A
```

**Comma-separated (all on one line):**
```
A, B, C, D, A, B, C, D, A, B
```

Both styles work. You can even mix them — MarkShark doesn't care about line breaks, commas, tabs, or spaces between answers.

---

## Multi-Version Keys

If your test has multiple versions, put a version header before each set of answers. The header tells MarkShark which version the answers belong to.

### Text file example
```
ver:A
A, B, C, D, A, B, C, D, A, B

ver:B
B, A, D, C, B, A, D, C, B, A
```

### CSV / TSV format
A tabular layout with one column per version, which is nice because you can see all versions side by side:
```
Q#,ver:A,ver:B,ver:C
1,A,B,C
2,B,A,D
3,C,D,A
4,D,C,B
5,A,B,C
```
- The `Q#` column is optional (MarkShark determines question order by row position).
- Column headers must contain `ver:` or `code:` to be detected as version columns.
- Blank rows and rows starting with `#` are skipped.

### Excel format (.xlsx)
Same tabular layout as CSV but in an Excel workbook. MarkShark reads the first data sheet and skips any sheets named "Instructions", "Help", or "ReadMe".

### Version labels must match the bubble sheet
Your version labels (the part after `ver:`) must match what the bubble sheet uses. If your bubble sheet has version bubbles labeled A, B, C, D then your key should use `ver:A`, `ver:B`, etc. If you accidentally use the wrong naming convention (for example `ver:1` when the sheet uses letters), MarkShark will detect the mismatch and offer to fix it automatically.

---

## Version Headers Explained

Each version header is a single line with one or more fields. Fields are case-insensitive.

| Field | Example | Purpose |
|-------|---------|---------|
| `ver:` | `ver:A` | Version letter — must match the bubble sheet's version labels |
| `code:` | `code:101` | Test code for machine matching (if your sheet has a test code zone) |
| `default:` | `default:2` | Points per question (defaults to 1 if omitted) |

You can combine fields on one line:
- `ver:A` — Version A, 1 point per question
- `ver:B default:3` — Version B, 3 points per question
- `ver:A code:101` — Version A with test code 101
- `code:201 default:2` — Test code 201, 2 points per question (no version letter)

At least one of `ver:` or `code:` is required for each version.

### Comments
Lines starting with `##` or `# ` (hash followed by a space) are ignored. Use them for notes:
```
## Biology 101 Midterm — January 2025
## Answer key created by Dr. Smith
ver:A
A, B, C, D, A
```

---

## Advanced Scoring Options

For most tests, each question has one correct answer worth one point. But MarkShark supports several other scoring modes for special situations.

### Custom point values
Add a colon and a number after any answer to override the default points:
```
A:3
```
Question is worth 3 points instead of the default.

### OR mode — accept multiple answers
Use `^` between answers when more than one answer should receive full credit:
```
A^B
```
A student who selects A **or** B gets full credit. Useful when you discover after the test that two answers are defensible.

### AND mode — require all answers
Use `&` between answers when the student must select all of them:
```
A&B
```
The student must bubble **both** A and B (exact match required).

### Partial credit (lenient)
Use `@` between answers. Correct selections earn points; wrong selections are ignored:
```
A@B
A:2@B:1
```
If the correct answers are A and B, a student who selects only A still gets partial credit. Selecting a wrong answer doesn't lose points, but filling more than two bubbles scores zero (spam protection).

### Partial credit (strict)
Use `~` between answers. Correct selections earn points; wrong selections lose points:
```
A~B
A:2~B:1
```
Same as lenient, but wrong selections subtract points. Still has spam protection.

### Freebie — everyone gets points
Use `*` when you want to give every student credit regardless of their answer:
```
*
*:5
```
Useful when you decide after the test that a question was unfair or had no correct answer.

### Discard — remove from scoring
Leave the answer blank (empty line in a text file, empty cell in a spreadsheet). The question is removed from the scoring denominator entirely, as if it didn't exist.

---

## Scoring Modes Summary

| Syntax | Mode | What happens |
|--------|------|--------------|
| `A` | Single | One correct answer, default points |
| `A:3` | Single | One correct answer, 3 points |
| `A^B` | OR | Either answer gets full credit |
| `A&B` | AND | Must select all listed answers |
| `A@B` | Partial (lenient) | +pts per correct, wrong ignored |
| `A~B` | Partial (strict) | +pts per correct, −pts per wrong |
| `*` | Freebie | Everyone gets points |
| *(blank)* | Discard | Question removed from scoring |

---

## Your Class Roster (optional)

A class roster helps MarkShark match scanned sheets to students by name and flag anyone who was absent. The roster is optional — without one, MarkShark still scores every sheet but can only identify students by whatever they bubbled in.

### Format
A CSV file (`.csv`) with at least these columns:

| Column | Also accepts |
|--------|-------------|
| **StudentID** | `ID`, `Student_ID`, `sid` |
| **LastName** | `Last`, `Surname`, `last_name` |

**FirstName** (`First`, `first_name`) is optional but recommended.

### Example
```
StudentID,LastName,FirstName
12345678,Smith,John
12345679,Johnson,Jane
12345680,Williams,Bob
```

Column detection is case-insensitive. Any extra columns (email, section number, etc.) are preserved but not used by MarkShark.

### Where to get your roster
Most learning management systems (Canvas, Brightspace, Moodle, etc.) can export a class list as CSV. Download it and MarkShark will usually recognize the columns automatically. If your LMS uses different column names, just rename them to match the table above.
