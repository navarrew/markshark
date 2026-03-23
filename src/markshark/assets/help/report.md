# Report

After scoring and reviewing corrections, the final step is to generate an Excel report. The report summarizes class performance, lists every student's score, provides item-by-item analysis, and consolidates any flags or issues onto a single sheet. You can share this report with colleagues, attach it to course records, or use it to improve future assessments.

## Table of Contents
- [How to Generate a Report](#how-to-generate-a-report)
- [Understanding the Report](#understanding-the-report)
- [The Summary Sheet](#the-summary-sheet)
- [Per-Version Sheets](#per-version-sheets)
- [The Class Scores Sheet](#the-class-scores-sheet)
- [The Answer Key Sheet](#the-answer-key-sheet)
- [The Flags & Issues Sheet](#the-flags--issues-sheet)
- [Simple Grade Mode](#simple-grade-mode)
- [Tips](#tips)

---

## How to Generate a Report

You can generate a report in two places:

- **Grader page** — after scoring, switch to the **Generate Report** tab.
- **Report Only page** — generates a report from an existing results CSV without re-scoring.

### What you need

- A **results CSV** (created during scoring). This is required.
- A **class roster** (optional). If provided, the report flags absent students and matches names to IDs.
- A **corrections file** (optional). If you made corrections on the Review & Correct page, they are applied automatically before the report is generated.

### Options

- **Assessment name** — a label for the report header (e.g. "Midterm 1").
- **Course name** — auto-filled from your Course Manager if available.
- **Simple Grade** — a checkbox that produces a streamlined report without item analysis. See [Simple Grade Mode](#simple-grade-mode) below.

Click **Generate Report** and MarkShark will create an `exam_report.xlsx` file in your assessment folder.

---

## Understanding the Report

The report is an Excel workbook with multiple sheets. Each sheet serves a different purpose:

| Sheet | What it contains |
|-------|-----------------|
| **Summary** | Class statistics, reliability, score distribution chart |
| **Version A, B, ...** | Per-student results and item analysis for each test version |
| **Class Scores** | Alphabetical student list with final scores (for gradebook import) |
| **Answer Key** | Correct answers and point values for each version |
| **Flags & Issues** | Absent students, flagged items, corrections, scoring parameters |

---

## The Summary Sheet

The first sheet gives you a bird's-eye view of the whole assessment.

### Class statistics

A table of statistics computed for each test version (and combined across all versions):

| Statistic | What it tells you |
|-----------|-------------------|
| **N Students** | Number of students who took this version |
| **Total Points** | Maximum possible score |
| **Mean** | Average score |
| **Mean %** | Average as a percentage of total points |
| **Median** | Middle score when all scores are ranked |
| **Std Dev** | How spread out the scores are. A small number means most students scored similarly; a large number means scores were widely spread. |
| **Min / Max / Range** | Lowest score, highest score, and the gap between them |
| **KR-20** | A reliability estimate for the test (0 to 1). Higher is better. A value above 0.70 suggests the test is reasonably reliable — students who know the material tend to do well across the whole test, not just on certain questions. Below 0.50 may indicate the test has too few questions or the questions aren't measuring the same thing. |
| **KR-21** | A more conservative reliability estimate. Usually slightly lower than KR-20. |

### Score distribution

A histogram chart showing how many students fall into each percentage-score range. The chart uses wider bins at the low end (0–10%, 10–20%, etc.) and finer 5% bins from 40% onward where the detail matters most.

### Flags & issues note

At the bottom, a one-line summary tells you whether any flags were raised (blanks, multi-marks, absent students, auto-detected versions) and points you to the Flags & Issues sheet for details. If everything is clean, you'll see a green checkmark.

---

## Per-Version Sheets

If your assessment has multiple versions (A, B, C, etc.), each version gets its own sheet with:

### Student results table

One row per student showing their name, ID, score breakdown (correct, incorrect, blank, multi), and their answer to every question.

### Item analysis

Below the student table, each question is analysed:

| Statistic | What it tells you |
|-----------|-------------------|
| **Item difficulty (%)** | The percentage of students who answered correctly. A question answered correctly by 95% of students is very easy; one answered by 20% is very hard. Aim for a mix — mostly in the 40–80% range. |
| **Point-biserial correlation** | How well each question separates strong students from weak ones (0 to 1). A high value (above 0.30) means students who did well on the test overall also tended to get this question right. A value near zero or negative means the question isn't doing a good job distinguishing — it might be confusing, ambiguous, or testing something different from the rest of the exam. |

Questions are colour-coded: green for good discrimination, yellow for moderate, and red for poor. This helps you quickly spot questions worth revising for next time.

---

## The Class Scores Sheet

An alphabetical list of all students with their final scores. Columns include:

- Last name, first name, student ID
- Score (number correct)
- Total possible points
- Percentage

This is the sheet you would typically use to enter grades into your LMS or gradebook. The [LMS Integration](utilities.md#lms-integration) page can also do this for you automatically.

---

## The Answer Key Sheet

Shows the correct answer for each question, organized by version. For each question you can see:

- The correct answer (or answers, for multi-answer questions)
- The point value (if weighted scoring was used)

This is useful for double-checking your key and for sharing the correct answers with students after the exam.

---

## The Flags & Issues Sheet

This sheet consolidates everything that needs attention into one place, keeping the Summary sheet focused on statistics.

### Absent students

If you provided a class roster, any students who appear on the roster but have no matching scanned sheet are listed here with their ID and name.

### Version auto-detected

If your test has multiple versions and some students didn't mark their version bubble, MarkShark assigns them to the version where they scored highest. This section shows those students, the version they were assigned, and their score on each version so you can verify the assignment is correct. The assigned version's score is highlighted in green.

### Flagged items

A table of students who had any issues during scoring:

| Column | What it shows |
|--------|---------------|
| **Student ID** | The student's bubbled ID |
| **Last Name / First Name** | Student name |
| **Version** | Test version |
| **Issues** | What was flagged — blanks, multi-fills, orphan IDs, auto-detected version, corrections applied |
| **Problem Questions** | Which questions had blank or multi-fill answers (up to 10 shown) |
| **Corrections Applied** | If you made corrections on the Review & Correct page, they appear here (e.g. "Q15: A→C") |

Issues are colour-coded: orange for orphan scans, blue for multi-fills, yellow for blanks, and a thin border for corrections.

### Scoring parameters

The scoring settings used for this run (Min Fill %, Fixed Threshold, Auto-Calibrate, etc.) are recorded here for reproducibility. If you ever need to replicate or troubleshoot a scoring run, these values tell you exactly what settings were in effect.

---

## Simple Grade Mode

Simple Grade mode produces a streamlined report designed for small classes or quick assessments where you don't need detailed item analysis. It is useful when you just want to know each student's score without the statistical deep-dive.

### What's included in a simple report

| Sheet | Included? |
|-------|-----------|
| Summary | Yes (lightweight — title and metadata only) |
| Per-version item analysis | No |
| Class Scores | Yes |
| Answer Key | Yes (answers and point values, but no difficulty/correlation statistics) |
| Flags & Issues | No |

### When to use simple grade

- Small classes (under ~30 students) where item statistics aren't meaningful
- Quick quizzes where you just need scores
- When you're using a single test version with no roster

You can enable simple grade mode with the checkbox on the Grader page. The setting is remembered per-assessment.

---

## Tips

- **Apply corrections first** — if you made corrections on the Review & Correct page, click **Apply Corrections & Re-annotate** before generating the report. The report uses the results CSV, so corrections need to be applied to that file first.
- **Check the Summary sheet first** — the class statistics and KR-20 give you a quick sense of how the test went overall before diving into individual questions.
- **Look for problem questions** — on the per-version sheets, questions highlighted in red have poor discrimination. These are worth reviewing: they might be poorly worded, have an incorrect key, or test something outside the course material.
- **Use Class Scores for your gradebook** — this sheet is designed for easy copy-paste or LMS import.
- **Re-generate freely** — generating a report doesn't change any of your scoring data. You can re-generate as many times as you like with different options.
