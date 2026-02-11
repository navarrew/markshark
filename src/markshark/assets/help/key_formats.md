# Answer Key Formats
MarkShark supports multiple answer key file formats. The key tells MarkShark which answers are correct for each question.
---
## Supported File Types
| Format | Extension | Notes |
|--------|-----------|-------|
| Text | `.txt` | Simplest format, one answer per line |
| CSV | `.csv` | Tabular, good for multi-version keys |
| TSV | `.tsv` | Tab-separated, same as CSV |
| Excel | `.xlsx` | Workbook with version columns |
---
## Text File Format (.txt)
The simplest format. Each line is the answer for the next question. Use `ver:` headers for multi-version keys.
### Single version (no header needed)
```
A
B
C
D
A
```
### Multi-version
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
### With test codes and point values
```
ver:A code:101 default:2
A
B:3
C
D
A
```
### Comma-separated (all answers on one line)
```
ver:A
A, B, C, D, A, B, C, D, A, B
```
### Comments
Lines starting with `##` or `# ` (hash followed by a space) are ignored.
```
## Biology 101 Midterm
## Created 2025-01-15
ver:A
A
B
# This is also a comment
C
```
---
## CSV / TSV Format
Tabular format with version columns. Good for multi-version keys where you want answers side-by-side.
```
Q#,ver:A,ver:B,ver:C
1,A,B,C
2,B,A,D
3,C,D,A
4,D,C,B
5,A,B,C
```
- The `Q#` column is optional (ignored by parser, question order is determined by row position).
- Column headers must contain `ver:` or `code:` to be detected.
- Empty rows, rows starting with `#`, `(`, `note`, or `example` are skipped.
---
## Excel Format (.xlsx)
Same tabular layout as CSV but in an Excel workbook.
- Sheets named "Instructions", "Help", or "ReadMe" are automatically skipped.
- The first data sheet is used.
- MarkShark can generate a blank template via the Grader page.
---
## Header Format
Version headers are case-insensitive and support these fields:
| Field | Aliases | Example | Required |
|-------|---------|---------|----------|
| Version letter | `ver:`, `version:` | `ver:A` | At least one of ver or code |
| Test code | `code:` | `code:101` | At least one of ver or code |
| Default points | `default:`, `defaultpoints:` | `default:2` | No (defaults to 1) |
**Examples:**
- `ver:A` — Version A, 1 point per question
- `ver:B default:3` — Version B, 3 points per question
- `ver:A code:101` — Version A with test code 101
- `code:201 default:2` — Test code 201, 2 points per question
---
## Answer Syntax
Each answer can use special operators for advanced scoring:
### Single answer (most common)
```
A
```
One correct answer, default points.
### Custom points
```
A:3
```
Answer A worth 3 points.
### OR mode — either answer accepted
```
A^B
A^B:4
```
Student gets full credit for selecting A **or** B.
### AND mode — must select all
```
A&B
A&B:4
```
Student must select **both** A and B (exact match required).
### Partial credit (lenient) — correct adds, wrong ignored
```
A@B
A:2@B:1
```
Each correct selection adds its points. Wrong selections are ignored. If a student fills more than 2 bubbles, they get 0 (spam protection).
### Partial credit (strict) — correct adds, wrong subtracts
```
A~B
A:2~B:1
```
Correct selections add points, wrong selections subtract points. Spam protection applies.
### Freebie — everyone gets points
```
*
*:5
```
All students receive the specified points regardless of their answer.
### Discard — remove from scoring
Leave the answer blank (empty line or empty cell). The question is removed from the scoring denominator entirely.
---
## Scoring Modes Summary
| Syntax | Mode | Behaviour |
|--------|------|-----------|
| `A` | Single | One correct answer |
| `A^B` | OR | Any one of the listed answers |
| `A&B` | AND | Must select all listed answers |
| `A@B` | Partial (lenient) | +pts correct, ignore wrong |
| `A~B` | Partial (strict) | +pts correct, -pts wrong |
| `*` | Freebie | Everyone gets points |
| *(blank)* | Discard | Question removed from scoring |
---
## Roster CSV Format
MarkShark can optionally use a class roster to identify students by name and flag absent students.
### Required columns
- **StudentID** (also accepts: `ID`, `Student_ID`, `sid`)
- **LastName** (also accepts: `Last`, `Surname`, `last_name`)
### Optional columns
- **FirstName** (also accepts: `First`, `first_name`)
### Example
```
StudentID,LastName,FirstName
12345678,Smith,John
12345679,Johnson,Jane
12345680,Williams,Bob
```
Column detection is case-insensitive. Additional columns are preserved but not used by MarkShark.
---
## Corrections Format
Corrections made on the Review & Correct page are stored in `score_data/corrections.csv`:
```
# applies to: /path/to/results.csv
timestamp,student_id,field,old_value,new_value
```
- Each edit appends a new row.
- A special `new_value` of `REVERT` undoes a previous correction.
- The effective correction for each student/field is the last non-REVERT entry.
