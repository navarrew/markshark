# How MarkShark Scores Your Bubble Sheets

This guide explains how MarkShark converts scanned bubble sheets into grades. Understanding this process will help you troubleshoot problems if something doesn't look right.

## The Basic Process

MarkShark takes a scanned PDF of completed bubble sheets and produces three outputs: a spreadsheet with student scores, colorized images showing how each bubble was read, and optional statistics about the exam. The scoring happens in stages, with each stage building on the previous one.

## Stage 1: Converting the Image to Black and White

When you scan a bubble sheet, you get a color or grayscale image with varying shades. MarkShark converts this into a pure black-and-white image using a **threshold**—a cutoff value that divides all the grays into just black or white. Imagine a slider from 0 (pure white) to 255 (pure black). Pixels darker than the threshold become black (ink); pixels lighter become white (paper). The default threshold is 127, right in the middle, but MarkShark can adjust this automatically per page if some students write very lightly.

**Why this matters:** If your scan is too light or too dark overall, MarkShark might misjudge which pixels are ink. Solution: scan at a consistent darkness, or let MarkShark's adaptive mode handle light marks.

## Stage 2: Identifying Bubble Locations

MarkShark uses a template file (called a "bubblemap") that tells it exactly where each bubble should be on the page. This template includes the position and size of every answer bubble, plus the name and ID fields. The bubblemap is defined as percentage distances from the edges, so it works with scans at different resolutions. Once MarkShark identifies the bubble locations, it creates small rectangular "regions of interest" (ROIs) around each bubble's center.

**Why this matters:** If your scanned pages are misaligned—rotated, skewed, or shifted—MarkShark won't be able to find the bubbles in the right places. MarkShark tries to auto-align pages using alignment markers, but if those fail, you may need to re-scan with better alignment.

## Stage 3: Measuring How Filled Each Bubble Is

For each bubble region, MarkShark measures what percentage of the bubble contains ink. This produces a "fill score" ranging from 0 (completely empty) to 1 (completely filled). The measurement focuses on a circular area inside each bubble rather than the entire rectangular region—this avoids counting stray marks just outside the bubble boundary.

**Example:** If a student filled in the "A" bubble halfway, that bubble gets a score of about 0.5. If they completely filled it in, it gets 1.0. If they didn't mark it at all, it gets close to 0.0.

**Why this matters:** Light pencil marks, smudges, and erasure remnants all contribute to the fill score. A student who barely tapped the bubble might get a score of 0.15 instead of a 0.0. This is where adaptive rescoring can help.

## Stage 4: Removing Background Noise

Bubble sheets are printed with faint circles marking where to fill in. These printed circles create a slight "background" signal even in empty bubbles. MarkShark measures this background by looking at the lowest-scoring bubbles in each column (usually empty ones). It computes the background level separately for each answer choice (A, B, C, D, E) because printed letters can be slightly darker. MarkShark then subtracts these background values from all the fill scores, giving a "cleaned" measurement that ignores the printed circles.

**Why this matters:** If your printed bubbles are very dark, the background subtraction becomes important. If they're very light, it matters less.

## Stage 5: Deciding Which Bubble Was Actually Marked

For each question row, MarkShark looks at the fill scores for all five choices (A through E) and picks the one with the highest score. But it only accepts this choice if certain conditions are met. The most important is that the highest score must be significantly higher than the second-highest. This prevents marking ambiguous rows as "answered" when the student circled two bubbles equally.

MarkShark uses three parameters to make this decision:

- **Minimum fill:** The highest score must be above this threshold (default: 0.15). If all scores are lower, the question is marked **blank**.
- **Top-2 ratio:** The second-highest score must be less than this percentage of the best score (default: 80%). If the second is too close, it's marked **multi** (multiple marks detected).
- **Minimum separation:** The difference between first and second must be above this percentage (default: 7 percentage points). This catches cases where both scores are very low but one is marginally higher.

**Example:** If a row's scores are [0.85, 0.12, 0.08, 0.05, 0.02], the "A" choice is clearly marked and would be selected. But if the scores are [0.30, 0.28, 0.05, 0.03, 0.02], even though "A" is highest, the second-place "B" is too close, so the question is marked **multi**.

## Stage 6: Adaptive Rescoring (For Light Marks)

If MarkShark detects blank questions even when the student appears to have marked bubbles lightly, it can use adaptive rescoring. This means it re-scans the page using a slightly higher threshold (making the system more sensitive to light marks). It tries multiple thresholds and picks the one that resolves the most blanks without creating new multi-mark problems. This happens automatically and is logged in the results.

**Why this matters:** Students who use very light pencils might have their marks missed on the first attempt. Adaptive rescoring can recover these automatically.

## Understanding the Output

### The Color-Coded Images
MarkShark creates annotated images where each bubble is outlined with a color showing how it was read:
- **Blue:** Name or ID field (informational only)
- **Green:** Correctly answered question
- **Red:** Incorrectly answered question
- **Gray:** Blank question (no mark detected)
- **Orange:** Multi-marked question (multiple bubbles detected)

If you see unexpected colors, these images show you exactly what MarkShark detected.

### The CSV Spreadsheet
The spreadsheet has columns for: student name, ID, version (if using multiple test versions), the number correct, the number incorrect, blanks, multi-marks, and percentage. If you provided an answer key, questions are scored as correct or incorrect.

If MarkShark detected blanks or multi-marks, it still counts them in the statistics but flags them (sometimes shown with an asterisk) so you can review them manually.

## Common Troubleshooting

**Many blanks detected:** Students likely used faint pencil marks. Try adjusting the minimum fill threshold lower (default 0.15 → try 0.10 or 0.05), or enable adaptive rescoring. Alternatively, ask students to mark more darkly.

**Many multi-marks detected:** Students may have circled multiple bubbles or the scan quality is poor. Check the color-coded images to see if the detected bubbles are real. If the scan is slightly misaligned, you might need to re-scan. If it's a consistent issue, you can adjust the top-2 ratio higher (default 0.80 → try 0.85).

**Misaligned pages:** If the images show bubbles in the wrong locations, the auto-alignment failed. Ensure your scan includes alignment markers (usually special codes on the page) or re-scan with the page placed straight.

**Some answers seem wrong:** Double-check the answer key file. A transposed version or misaligned key can cause widespread errors.

**Inconsistent results across pages:** Different paper quality, scanner settings, or lighting during scanning can cause variation. Ensure your entire batch of scans was taken under consistent conditions.

## Summary

MarkShark's scoring is a multi-step process that handles real-world imperfections in scanning and hand-marking. By understanding how the stages work together—binarization, alignment, fill measurement, background removal, and bubble selection—you can make informed adjustments if needed. The color-coded output images are your best friend for understanding what the system detected, and the adaptive rescoring option can help recover lightly-marked answers.
