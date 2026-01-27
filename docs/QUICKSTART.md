# MarkShark Quickstart

This guide focuses on the first successful end to end run, plus the most common workflows.

## Install

Create a fresh environment, then install:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows PowerShell

pip install -U pip
pip install markshark
```

Verify the install:

```bash
markshark --help
markshark templates --help
```

## Typical workflow

MarkShark is usually run in one of these two ways:

1. Two step pipeline, align then score.
2. One step pipeline, `quick_grade`, which runs align then score using a named template.

### Two step pipeline

1) Align scans to a template PDF:

```bash
markshark align scans.pdf --template template.pdf --out-pdf aligned_scans.pdf
```

2) Score the aligned scans using a bubblemap and optional answer key:

```bash
markshark score aligned_scans.pdf --bublmap bubblemap.yaml --out-csv results.csv
# With an answer key
markshark score aligned_scans.pdf --bublmap bubblemap.yaml --key-txt key.txt --out-csv results.csv
```

Common outputs:

- `results.csv` (per student results)
- annotated PDF, if `--out-pdf` is enabled or defaults apply
- annotated PNG directory, if `--out-annotated-dir` is provided

### One step pipeline with templates

List available templates:

```bash
markshark templates
```

Then run:

```bash
markshark quick_grade scans.pdf --template TEMPLATE_ID --key-txt key.txt --out-csv results.csv
```

## Debugging and verification

Visualize your bubblemap overlay (sanity check zones and numbering):

```bash
markshark visualize template.pdf --bublmap bubblemap.yaml --page 1 --out-image bubblemap_overlay.png
```

If you are tuning scoring thresholds, the most relevant flags are:

- `--min-fill`
- `--top2-ratio`
- `--min-top2-diff`
- `--fixed-thresh` and `--auto-thresh/--no-auto-thresh`

You can also generate focused review artifacts:

- `--review-pdf` to output only pages with flagged answers
- `--flagged-xlsx` to list flagged items for manual review

## Generate an Excel report

```bash
markshark report results.csv --out-xlsx exam_report.xlsx
```

## Launch the GUI

```bash
markshark gui --port 8501
# or
markshark-gui
```

## Full CLI reference

See `CLI_REFERENCE.md` for the complete list of commands and options.
