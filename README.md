# MarkSharkOMR
## A fast, accurate, customizable, open-source test bubble sheet scanner

MarkSharkOMR is a versatile and fast tool to **visualize, grade, and analyze bubble-sheet exams**.

It supports command-line and GUI modes, and outputs annotated images, CSV results, and detailed item-analysis statistics.


![logo](images/shark.png)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

---

![OMR pipeline](images/processmap.png)

---

## Prerequisites

- **Python 3.9–3.12**

---

## Installation

Clone and install the package locally:

```bash
git clone https://github.com/navarrew/markshark.git
cd markshark
pip install -e .
```

Make sure you install required dependencies (**OpenCV**, **Typer**, **Streamlit**).

---

# Using MarkShark

## Step 1 – Make Your Master Bubble Sheet

### Create your bubble sheet (the exam form)

Start by making a bubble sheet and saving it as a **PDF file**.

- You can use the ready-made templates (and their corresponding config files) in the **`templates`** folder.  
- You can freely modify these templates for your own use.  
- To design your own sheet, use any drawing or layout program that exports to PDF.  
  - **Affinity Designer** (similar to Adobe Illustrator) works well.  
  - **Inkscape** is a great free option that supports `.svg` files.  
  - Templates are provided in both PDF and SVG formats for easy editing.


### (Optional) Add alignment markers (ARUCO markers)

We recommend adding **ARUCO markers**—small black-and-white squares in the corners of each page.

- These act as landmarks that help the software detect and correct **rotation**, **skew**, or **scaling** in scanned images.  
- PNG marker files are included in the template folder; paste one in each corner of your sheet.  
- You can also generate ARUCO markers using free online tools or MarkShark’s utilities.  
- Without them, even slight scan misalignment can reduce grading accuracy.

---

## Step 2 – Make a bubblemap file for your bubble sheet

The bubblemap that tells MarkShark where the important areas are located on your sheet including:
- student ID bubbles  
- answer bubbles  
- name boxes, etc.

The configuration file is in YAML format and you only need to make it once and can reuse it indefinitely for a given type of bubble sheet.  However, if you change where the bubbles are located on your bubble sheet you will have to update your configuration file to match.

> **Tip:** It's worth putting in some time to make sure your map fits well to your master bubble sheet.  If the zones in the config file don’t line up with the printed bubbles, grading will fail or be inaccurate.

**To check and adjust your config file:**
- Use MarkShark’s **visualizer** function to preview how your `bubblemap.yaml` zones align with your template.  
- Adjust coordinates and re-test until everything matches cleanly.

---

## Step 3 - Make your answer key

Create a plain-text file (e.g. `key.txt`) listing the correct answers.

- Answers can be separated by **commas** (`A,B,A,D,C,C,D`)  
  or placed on **separate lines** (one per question).  
- The key can be shorter than the total number of bubbles.  Unused ones will simply be ignored.

---

## Step 4 – Scan and Align Your Bubble Sheets

You can scan student sheets using any standard desktop scanner (flatbed or sheet-fed).

Even high-quality scans are often slightly misaligned. Pre-aligning them ensures precise grading.

1. **Scan the sheets**  
   - Use a high-quality scanner.  
   - Avoid phone photos—uneven lighting and perspective distortions are difficult to correct.

2. **Export scans**  
   - Save them as a **single multi-page PDF**, at **150 dpi** or **300 dpi**, grayscale or color.

3. **Align the scans**  
   - Use the MarkShark **`align`** command to align scanned pages with the clean template PDF.  
   - The output will be a new, perfectly aligned multipage PDF for grading.

---

## Step 5 – Score

1. Run the **`score`** function on the aligned PDF in a directory containing your `key.txt` and `config.yaml`.  
2. MarkShark outputs a **CSV file** with each student’s answers and final score.  
3. Optionally, export an **annotated PDF** showing which bubbles were recognized.  
   - This is useful for debugging alignment or fill-threshold issues.

---

## Step 6 – Analyze Results

1. Run the **`stats`** function on the CSV output from the grading step.  
2. This generates a report summarizing overall performance, item difficulty, and question quality.  

---

# Command Examples

*(Add usage snippets here — align, grade, and stats commands as examples.)*

---

# Additional Instructions

## Designing Your Bubble Sheet

1. Align bubbles with **consistent spacing** in rows and columns.  
   Use “align” and “distribute” features in your design software.  
2. Use **gray** or **semi-transparent black** circles so student marks stand out.  
   (Our templates use hollow black circles at 50 % opacity.)

---

## Using the Visualizer to Build Your Bubblemap File

Bubblemap files use **YAML** format.  
The visualizer overlays your configuration zones on the sheet, helping you fine-tune coordinates.

---

## Aligning Your Scans

> “Garbage in, garbage out.”  
> Use the highest-quality scans possible (300 dpi recommended).  
> Avoid scanners that warp or compress pages unevenly.

---

## Scoring Tests with `grade`

Once aligned, grading is automatic.  
All results are written to a CSV file, ready for analysis.

---

## Analyzing Results with `stats`

The **`stats`** module processes the CSV output from `score`, computes exam- and item-level metrics, and generates CSV summaries and plots.

### Usage

```bash
markshark stats results.csv
```

### Outputs

1. **Main CSV** – `results_with_item_stats.csv`  
   - Original data (including KEY row)  
   - Adds rows for:
     - *Pct correct (0–1)*: item difficulty  
     - *Point–biserial*: discrimination index  

2. **Exam stats** – `exam_stats.csv`  
   - Overall test summary:
     - `k_items`: number of questions  
     - `mean_total`, `sd_total`, `var_total`: score distribution  
     - `avg_difficulty`: mean proportion correct  
     - `KR-20`, `KR-21`: reliability estimates  

3. **Item report** – `item_analysis.csv`  
   - One row per item-option:
     - `item`, `key`, `option`, `is_key`  
     - `option_count`, `option_prop`  
     - `option_biserial`, `item_difficulty`, `item_point_biserial`

4. **Plots** – folder `item_plots/`  
   - One PNG per item showing Item Characteristic Curves (ICC):  
     - **X-axis:** binned total-minus-item score  
     - **Y-axis:** proportion correct per bin

---

### Interpretation of Statistics

| Metric | Description | Ideal / Notes |
|:--|:--|:--|
| **Difficulty (Pct correct)** | Proportion of students answering correctly. | Ideal range ≈ 0.3–0.8 |
| **Point–biserial** | Correlation between item correctness and total score (excluding that item). | Higher = better discrimination; negative = problematic |
| **KR-20** | Reliability coefficient for dichotomous items. | ≥ 0.7 = good internal consistency |
| **KR-21** | Approximation of KR-20 assuming equal item difficulty. | Use when item data are limited |
| **Option biserial** | Correlation between selecting an option and student ability. | Correct = positive; distractors = negative / ~0 |

---

## Outputs Summary

- **Aligned scans:** `aligned_scans.pdf`  
- **Zone overlay visualization:** PNG previews  
- **Grading results:** `results.csv`  
- **Annotated sheets:** optional PNGs in `annotated/`  
- **Statistical analysis:** KR-20 reliability, difficulty, point-biserial, and ICC plots

---

## License

**MarkShark — the open-source bubble hunter**  
Copyright © 2026 William Navarre, University of Toronto  
Licensed under the **GNU Affero General Public License v3 (AGPL-3.0)**.

You may use, modify, and redistribute this software for **academic, research, and open-source** purposes, provided derivative works remain open-source under the same license.

**Commercial or institutional use** (e.g., SaaS platforms, proprietary educational tools, or for-profit distribution) requires a separate license.

For licensing inquiries, contact [william.navarre@utoronto.ca](mailto:william.navarre@utoronto.ca).
