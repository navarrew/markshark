# MarkShark CLI Reference

Generated from the repository `cli.py` and `defaults.py` as of 2026-01-26.

## Installed entry points

- `markshark` -> `markshark.cli:app_main`
- `markshark-streamlit` -> `markshark.streamlit_launcher:main` (Streamlit web interface)
- `python -m markshark.gui` (Native PySide6 desktop GUI)

## Global help

- `markshark --help`
- `markshark <command> --help`

## Commands

## `align`

Align raw scans to a template PDF.

ALIGNMENT METHODS:

- auto: Uses 'fast' if --bubblemap provided, else 'slow' (recommended)
- fast: Coarse-to-fine alignment. Quick 72 DPI ORB pass, then bubble grid
        refinement at full res. Requires --bubblemap. Best for bubble sheets.
- slow: Full resolution ORB alignment. More thorough but slower.
        Works without bubblemap.
- aruco: ArUco marker alignment only. Requires markers on the sheet.

**Usage**

`markshark align <input_pdf> [OPTIONS]`

**Arguments**
- `input_pdf` (str, required). Raw scans PDF

**Options**
- `--template`, `-t` sets `template` (str). Default: required. Template PDF to align to
- `--out-pdf`, `-o` sets `out_pdf` (str). Default: 'aligned_scans.pdf'. Output aligned PDF
- `--dpi` sets `dpi` (int). Default: 150. Render DPI for alignment & output
- `--template-page` sets `template_page` (int). Default: 1. Template page index to use (1-based)
- `--align-method` sets `align_method` (str). Default: 'auto'. Alignment method: auto|fast|slow|aruco. fast=coarse-to-fine (72 DPI ORB + bubble grid, requires --bubblemap), slow=full-res ORB only, auto=fast if bubblemap provided else slow
- `--estimator-method` sets `estimator_method` (str). Default: 'auto'. Homography estimator: auto|ransac|usac
- `--min-markers` sets `min_markers` (int). Default: 4. Min ArUco markers to accept
- `--ransac` sets `ransac` (float). Default: 3.0. RANSAC reprojection threshold
- `--use-ecc/--no-use-ecc` sets `use_ecc` (bool). Default: True. Enable ECC refinement
- `--ecc-max-iters` sets `ecc_max_iters` (int). Default: 50. ECC iterations
- `--ecc-eps` sets `ecc_eps` (float). Default: 1e-06. ECC termination epsilon
- `--orb-nfeatures` sets `orb_nfeatures` (int). Default: 3000. ORB features for feature-based align
- `--match-ratio` sets `match_ratio` (float). Default: 0.75. Lowe's ratio test for feature matching
- `--dict-name` sets `dict_name` (str). Default: 'DICT_4X4_50'. ArUco dictionary
- `--first-page` sets `first_page` (Optional[int]). Default: None. First page index (1-based)
- `--last-page` sets `last_page` (Optional[int]). Default: None. Last page index (inclusive, 1-based)
- `--bubblemap`, `-m` sets `bubblemap_path` (Optional[str]). Default: None. Bubblemap YAML file. Enables 'fast' alignment mode (coarse-to-fine with bubble grid).

## `mapviewer`

Overlay the bublmap bubble zones on top of a PDF page to verify placement.

Use this to visualize where MarkShark expects to find bubbles on your template.

**Usage**

`markshark mapviewer <input_pdf> [OPTIONS]`

**Arguments**
- `input_pdf` (str, required). An aligned page PDF or template PDF

**Options**
- `--bublmap`, `-m` sets `bublmap` (str). Default: required. Bubblemap file (.yaml/.yml)
- `--out-image`, `-o` sets `out_image` (str). Default: 'bubblemap_overlay.png'. Output overlay image (png/jpg/pdf)
- `--pdf-renderer` sets `pdf_renderer` (str). Default: 'auto'. PDF renderer: auto|fitz|pdf2image
- `--dpi` sets `dpi` (int). Default: 150. Render DPI

## `mock-dataset`

Generate a mock dataset from a template for testing.

Creates synthetic student scans with filled bubbles, an answer key,
and a CSV with expected student responses.

**Usage**

`markshark mock-dataset [OPTIONS]`

**Options**
- `--template`, `-t` sets `template_id` (str). Default: required. Template ID or display name
- `--out-dir`, `-o` sets `out_dir` (str). Default: required. Output directory for generated files
- `--num-students`, `-n` sets `num_students` (int). Default: 100. Number of fake students to generate
- `--seed` sets `seed` (int). Default: 42. Random seed for reproducibility
- `--dpi` sets `dpi` (int). Default: 300. DPI for rendered images
- `--templates-dir` sets `templates_dir` (Optional[str]). Default: None. Custom templates directory
- `--darkness-min` sets `darkness_min` (float). Default: 0.4. Minimum bubble darkness (0-1)
- `--darkness-max` sets `darkness_max` (float). Default: 1.0. Maximum bubble darkness (0-1)
- `--apply-transform` sets `apply_transform` (bool). Default: False. Apply random rotation/translation
- `--blank-rate` sets `blank_rate` (float). Default: 0.02. Rate of blank answers
- `--multi-rate` sets `multi_rate` (float). Default: 0.02. Rate of multi-fill answers

## `score`

Grade aligned scans using axis-based bublmap.

When --key-txt is provided and --include-stats is enabled (default), the output CSV
will include summary rows at the bottom with:
- Exam statistics: N, Mean, StdDev, High/Low scores, KR-20 reliability
- Item statistics: % correct and point-biserial for each question

**Usage**

`markshark score <input_pdf> [OPTIONS]`

**Arguments**
- `input_pdf` (str, required). Aligned scans PDF

**Options**
- `--bublmap`, `-c` sets `bublmap` (str). Default: required. Bubblemap file (.yaml/.yml)
- `--key-txt`, `-k` sets `key_txt` (Optional[str]). Default: None. Answer key file (A/B/C/... one per line). If provided, only first len(key) questions are scored.
- `--out-csv`, `-o` sets `out_csv` (str). Default: 'results.csv'. Output CSV of per-student results
- `--out-annotated-dir` sets `out_annotated_dir` (Optional[str]). Default: None. Directory to write annotated sheets
- `--out-pdf` sets `out_pdf` (Optional[str]). Default: uses defaults ('scored_scans.pdf'). Annotated PDF output filename. Default: scored_scans.pdf. Use """ to disable.
- `--review-pdf` sets `review_pdf` (Optional[str]). Default: None. Output PDF containing only pages with flagged answers (blank/multi). Use for manual review.
- `--flagged-xlsx` sets `flagged_xlsx` (Optional[str]). Default: None. Output XLSX listing flagged items (blank/multi) with Corrected Answer column for manual review.
- `--annotate-all-cells` sets `annotate_all_cells` (bool). Default: False. Draw every bubble in each row
- `--label-density` sets `label_density` (bool). Default: False. Overlay % fill text at bubble centers
- `--dpi` sets `dpi` (int). Default: 150. Scan/PDF render DPI
- `--min-fill` sets `min_fill` (Optional[float]). Default: uses defaults (0.45). Minimum fraction of the darkest bubble required to consider a mark filled (default: 0.45).
        Increase to require more completely filled bubbles; decrease to accept lighter or partially filled marks.
- `--top2-ratio` sets `top2_ratio` (Optional[float]). Default: uses defaults (0.8). default 0.8
- `--min-top2-diff` sets `min_top2_diff` (Optional[float]). Default: uses defaults (10.0). Minimum difference (in percentage points) between top 2 bubbles to not score as multi (default: 10.0).
        Increase to require larger separation; decrease to accept closer scores.
- `--fixed-thresh` sets `fixed_thresh` (Optional[int]). Default: uses defaults (180). default 180
- `--auto-thresh/--no-auto-thresh` sets `auto_thresh` (bool). Default: True. Auto tune fixed_thresh per page when --fixed-thresh is omitted
- `--verbose-thresh` sets `verbose_calibration` (bool). Default: False. Print per-page threshold calibration diagnostics
- `--include-stats/--no-include-stats` sets `include_stats` (bool). Default: True. Append basic statistics (mean, std, KR-20, item difficulty, point-biserial) to CSV. Requires answer key.

## `report`

Generate an Excel report with per-version tabs, item analysis, and roster checking.

The report includes:
- Summary tab with overall exam statistics
- Per-version tabs with student results and item statistics
- Roster matching (if --roster provided): flags absent students and orphan scans
- Color-coded item quality indicators
- Project metadata (if --project-name or --run-label provided)

**Usage**

`markshark report <input_csv> [OPTIONS]`

**Arguments**
- `input_csv` (str, required). Results CSV from 'score'

**Options**
- `--out-xlsx`, `-o` sets `out_xlsx` (str). Default: 'exam_report.xlsx'. Output Excel report
- `--roster`, `-r` sets `roster_csv` (Optional[str]). Default: None. Optional class roster CSV (StudentID, LastName, FirstName)
- `--project-name` sets `project_name` (Optional[str]). Default: None. Project name to include in report header
- `--run-label` sets `run_label` (Optional[str]). Default: None. Run label (e.g., run_001_2025-01-21_1430) to include in report header

## `templates`

List available bubble sheet templates.

**Usage**

`markshark templates  [OPTIONS]`

**Options**
- `--templates-dir`, `-d` sets `templates_dir` (Optional[str]). Default: None. Templates directory (default: auto-detect)
- `--validate`, `-v` sets `validate` (bool). Default: False. Validate each template

## `quick_grade`

Quick grade: align + score in one command using a template.

This command automatically uses bubble grid alignment as a fallback when
ArUco markers are not detected, using the bubble positions from the template's
bubblemap YAML.

**Usage**

`markshark quick_grade <input_pdf> [OPTIONS]`

**Arguments**
- `input_pdf` (str, required). Raw student scans PDF

**Options**
- `--template`, `-t` sets `template_id` (str). Default: required. Template ID or display name (use 'markshark templates' to list)
- `--key-txt`, `-k` sets `key_txt` (Optional[str]). Default: None. Answer key file (optional)
- `--out-csv`, `-o` sets `out_csv` (str). Default: 'quick_grade_results.csv'. Output CSV of results
- `--out-pdf` sets `out_pdf` (str). Default: 'quick_grade_annotated.pdf'. Output annotated PDF
- `--out-dir` sets `out_dir` (Optional[str]). Default: None. Output directory (default: same as out_csv)
- `--dpi` sets `dpi` (int). Default: 150. Render DPI
- `--templates-dir` sets `templates_dir` (Optional[str]). Default: None. Custom templates directory
- `--align-method` sets `align_method` (str). Default: 'auto'. Alignment method: auto|aruco|feature
- `--min-markers` sets `min_markers` (int). Default: 4. Min ArUco markers to accept
- `--min-fill` sets `min_fill` (Optional[float]). Default: uses defaults (0.45). Min fill threshold (default: 0.45)
- `--top2-ratio` sets `top2_ratio` (Optional[float]). Default: uses defaults (0.8). Top2 ratio (default: 0.8)
- `--min-top2-diff` sets `min_top2_diff` (Optional[float]). Default: uses defaults (10.0). Min difference between top 2 bubbles (default: 10.0)
- `--annotate-all-cells` sets `annotate_all_cells` (bool). Default: False. Draw every bubble in each row
- `--label-density` sets `label_density` (bool). Default: False. Overlay % fill text
- `--auto-thresh/--no-auto-thresh` sets `auto_thresh` (bool). Default: True. Auto-calibrate threshold

## `streamlit`

Launch the Streamlit web interface.

For the native desktop GUI, use: `python -m markshark.gui`

**Usage**

`markshark streamlit [OPTIONS]`

**Options**
- `--port` sets `port` (int). Default: auto-detected. Port to serve Streamlit web interface
- `--headless` Run without opening a browser (for remote servers)
