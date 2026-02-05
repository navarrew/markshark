# MarkShark Answer Key Format Guide

MarkShark supports flexible answer key formats that go far beyond simple single-letter answers. This guide explains all supported formats and scoring modes.

## Supported File Types

| Extension | Description |
|-----------|-------------|
| `.txt` | Plain text file (single or multi-version) |
| `.csv` | Comma-separated tables of values |
| `.tsv` | Tab-separated tables of values |
| `.xlsx` | Excel workbook (recommended for complex keys) |

## Quick Start

### Simple Text File (Legacy Format)
For basic exams with a single version and single simple correct answers worth 1 point each:
```
A
B
C
A
D
```
Each line is one question. This format assumes Version A and 1 point per question.

### Modern Text File Format
for more complex scenarios.  
Here you have a test with multiple versions and where each question is worth 2 points unless specified otherwise.
```
# Chemistry Final Exam
ver:A default:2
A
B:4
C^D
A@B
*
B
```
In the example above:
- the answer to the first question is 'A' and worth the default 2 points
- The answer to the second question is 'B' and worth 4 points 
- for question 3 either C OR D are acceptable answers


### Excel/CSV Format
| Q# | ver:A default:2 | ver:B | code:101 |
|----|-----------------|-------|----------|
| 1  | A               | B     | C        |
| 2  | B:4             | A:4   | D:4      |
| 3  | C^D             | D^E   | A^B      |
| 4  | A@B             | B@C   | A@D      |
| 5  | *               | *:3   | *        |
| 6  | B               | A     | C        |

---

## Header Format

Each version/test code needs a header. Headers are **case-insensitive**.

### Required: Version or Code (at least one)
```
ver:A                    # Version A
ver:B                    # Version B
code:101                 # Test code 101
code:202                 # Test code 202
```

### Optional: Default Points
```
ver:A default:2          # Version A, 2 points per question
code:101 default:3       # Code 101, 3 points per question
```

### Combined
```
ver:A code:101 default:2  # Version A with code 101, 2 pts/question
```

**Note:** If no `default:` is specified, questions are worth 1 point each.

---

## Answer Formats

### Single Answer
The simplest format - one correct answer.

| Format | Meaning | Points |
|--------|---------|--------|
| `A` | Answer is A | default |
| `B:3` | Answer is B | 3 points |
| `C:0.5` | Answer is C | 0.5 points |

**Example:** Student answers A when key is `A` → full credit

---

### OR Mode (`^`)
Multiple acceptable answers - student picks ONE, any correct answer gets full credit.

| Format | Meaning | Points |
|--------|---------|--------|
| `A^B` | A or B accepted | default |
| `A^B^C` | A, B, or C accepted | default |
| `A^B:4` | A or B accepted | 4 points |

**Use case:** When a question has multiple defensible answers, or you realize after printing that two answers could be correct.

**Scoring:**
- Student answers A when key is `A^B` → full credit
- Student answers B when key is `A^B` → full credit
- Student answers C when key is `A^B` → 0 points
- Student answers A,B (multi-bubble) when key is `A^B` → 0 points (multi-mark penalty)

---

### AND Mode (`&`)
Student must select ALL specified answers exactly - no more, no less.

| Format | Meaning | Points |
|--------|---------|--------|
| `A&B` | Must select both A and B | default |
| `A&B&C` | Must select A, B, and C | default |
| `A&B:4` | Must select both A and B | 4 points |

**Use case:** "Which TWO of the following are correct?" where you want exact matching only.

**Scoring:**
- Student answers A,B when key is `A&B` → full credit
- Student answers B,A when key is `A&B` → full credit (order doesn't matter)
- Student answers A when key is `A&B` → 0 points (missing B)
- Student answers A,B,C when key is `A&B` → 0 points (extra answer)

---

### Partial Credit - Lenient (`@`)
Award points for each correct answer. Wrong answers are ignored. Anti-spam protection applies.

| Format | Meaning | Points |
|--------|---------|--------|
| `A@B` | +default for A, +default for B | sum of parts |
| `A:2@B:1` | +2 for A, +1 for B | max 3 points |
| `A@B@C` | +default for each correct | sum of parts |

**Use case:** "Select all that apply" where you want to reward partial knowledge without punishing guessing.

**Scoring (key: `A@B` with default:1):**
- Student answers A,B → 2 points (full credit)
- Student answers A → 1 point (partial)
- Student answers B → 1 point (partial)
- Student answers A,C → 1 point (A correct, C ignored)
- Student answers A,B,C → 0 points (SPAM: 3 answers > 2 correct options)

**Anti-spam rule:** If student fills more bubbles than there are correct answers, they get 0.

---

### Partial Credit - Strict (`~`)
Award points for correct answers, SUBTRACT points for wrong answers. Anti-spam protection applies.

| Format | Meaning | Points |
|--------|---------|--------|
| `A~B` | +default for correct, -default for wrong | sum of parts |
| `A:2~B:1` | +2 for A, +1 for B, penalty for wrong | max 3 points |

**Use case:** "Select all that apply" where you want to discourage guessing.

**Scoring (key: `A~B` with default:1):**
- Student answers A,B → 2 points (full credit)
- Student answers A → 1 point (partial)
- Student answers A,C → 0 points (A=+1, C=-1, floor at 0)
- Student answers C,D → 0 points (both wrong, floor at 0)
- Student answers A,B,C → 0 points (SPAM)

**Note:** Points cannot go negative - minimum is 0 per question.

---

### Freebie (`*`)
Everyone gets full credit regardless of their answer.

| Format | Meaning | Points |
|--------|---------|--------|
| `*` | Everyone gets default points | default |
| `*:3` | Everyone gets 3 points | 3 points |

**Use case:** Question was ambiguous, flawed, or you want to give everyone credit for a bonus question.

**Scoring:**
- Any answer (including blank) → full points
- Question still counts toward the denominator

---

### Discard (blank)
Remove the question entirely from scoring.

| Format | Meaning |
|--------|---------|
| (empty cell) | Question doesn't exist |
| (whitespace only) | Question doesn't exist |

**Use case:** Question was fundamentally broken and shouldn't count at all.

**Scoring:**
- Question is removed from both numerator AND denominator
- As if the question never existed

---

## Important Notes for Teachers

### Partial Credit Questions Require Clear Instructions

When using `@` or `~` (partial credit), your question **MUST** tell students how many answers to select:

**Good:** "Select the TWO correct answers from the following:"
**Bad:** "Select all that apply" (students don't know how many)

The anti-spam rule (more bubbles than correct answers = 0) requires students to know the expected count.

### Choosing Between Partial Credit Modes

| Mode | Wrong answers | Best for |
|------|---------------|----------|
| `@` (lenient) | Ignored | Encouraging partial knowledge |
| `~` (strict) | Subtract points | Discouraging guessing |

### Multi-Version Exams

When creating multiple versions, ensure:
1. Same number of questions across all versions
2. Question difficulty is balanced
3. Each version has its own header line

```
ver:A default:1
A,B,C,D,A,B,C,D,A,B

ver:B default:1
B,C,D,A,B,C,D,A,B,C
```

### Test Codes vs Versions

- **Versions** (ver:A, ver:B): Letters bubbled on the answer sheet
- **Test Codes** (code:101): Numbers bubbled in the test ID field

You can use either or both. MarkShark matches students to keys by:
1. Version letter (if provided)
2. Test code (if provided)
3. Best match by score (fallback)

---

## File Format Examples

### Text File - Single Version
```
# Biology Midterm
ver:A default:2
A
B:4
C^D
A@B
*
B
```

### Text File - Multi-Version
```
# Chemistry Final
ver:A default:1
A,B,C,D,A,B,C,D,A,B

ver:B default:1
B,C,D,A,B,C,D,A,B,C

ver:C default:1
C,D,A,B,C,D,A,B,C,D
```

### CSV File
```csv
Q#,ver:A default:2,ver:B,code:101
1,A,B,C
2,B:4,A:4,D:4
3,C^D,D^E,A^B
4,A@B,B@C,A@D
5,*,*:3,*
6,B,A,C
```

### Excel File
Use the template provided in the MarkShark GUI (Quick Grade → Download answer key template).

The template includes:
- **Instructions tab** with full format documentation
- **Answer Key tab** pre-formatted with Q# formulas and 4 version columns

---

## Scoring Summary Table

| Format | Example | Student: A | Student: B | Student: A,B | Student: A,B,C |
|--------|---------|------------|------------|--------------|----------------|
| Single | `A` | ✓ Full | ✗ 0 | ✗ 0 (multi) | ✗ 0 (multi) |
| OR | `A^B` | ✓ Full | ✓ Full | ✗ 0 (multi) | ✗ 0 (multi) |
| AND | `A&B` | ✗ 0 | ✗ 0 | ✓ Full | ✗ 0 |
| Partial@ | `A@B` | ½ | ½ | ✓ Full | ✗ 0 (spam) |
| Partial~ | `A~B` | ½ | ½ | ✓ Full | ✗ 0 (spam) |
| Freebie | `*` | ✓ Full | ✓ Full | ✓ Full | ✓ Full |

---

## Troubleshooting

### "Advanced key format detected" message
This confirms MarkShark recognized your key uses the new format features (ver:, ^, &, @, ~, or point overrides).

### Students getting 0 on partial credit questions
Check if they're hitting the anti-spam rule. If key is `A@B` (2 correct), any student bubbling 3+ answers gets 0.

### Version mismatch warnings
If a student's bubbled version doesn't match any key, MarkShark will:
1. Try to match by test code
2. Score against all versions and use the best match
3. Mark the version with `*` (e.g., "A*") to indicate auto-detection

### Legacy keys not working
Ensure your legacy `.txt` file has one answer per line with no headers. Or convert to the new format by adding `ver:A` at the top.
