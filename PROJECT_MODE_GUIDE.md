# MarkShark Project Mode Guide

## Overview

MarkShark now supports **Project Mode** - a structured file organization system that helps you manage exam grading workflows with clear naming, versioning, and a predictable file structure.

## What's New?

### Before (Temporary Mode)
- Files saved to random temporary directories: `score_xkdj3k2`, `align_9dk2jd`
- No connection between input files and output files
- Easy to lose track of which files belong to which exam
- Reports don't show context about what exam they're for

### After (Project Mode)
- Organized project folders: `FINAL_EXAM_BIO101_2025/`
- Versioned runs: `run_001_2026-01-21_1430/`
- Input files saved for reference
- Reports include project name and timestamp
- Easy to find all files related to a specific exam

## Directory Structure

When you use Project Mode, MarkShark creates this structure:

```
{Your Working Directory}/
└── {PROJECT_NAME}/
    ├── input/                    # Original uploaded files
    │   ├── scans_001.pdf
    │   ├── answer_key_001.txt
    │   └── ...
    ├── aligned/                  # Aligned PDFs
    │   ├── aligned_scans_001.pdf
    │   ├── aligned_scans_002.pdf
    │   └── ...
    ├── results/                  # Versioned scoring runs
    │   ├── run_001_2026-01-21_1430/
    │   │   ├── results.csv
    │   │   ├── scored_scans.pdf
    │   │   ├── for_review.pdf
    │   │   ├── flagged.csv
    │   │   ├── exam_report.xlsx
    │   │   └── annotated_pngs/
    │   ├── run_002_2026-01-21_1545/
    │   │   └── ...
    │   └── run_003_2026-01-22_0900/
    │       └── ...
    └── config/                   # Configuration files
        ├── template_used.txt
        └── bubblemap_overlay.png
```

## How to Use Project Mode

### GUI (Streamlit)

1. **Set Working Directory**
   - In the sidebar, navigate to your desired working directory
   - Example: `~/Desktop/Grading/`

2. **Enter Project Name**
   - In the sidebar under "Current Project", enter a descriptive name
   - Example: `FINAL EXAM BIO101 2025`
   - The folder name will be automatically sanitized: `FINAL_EXAM_BIO101_2025`

3. **Run Your Workflow**
   - Use Quick Grade, Align, Score, or Report as normal
   - Files will automatically be saved to the project structure
   - Each scoring run gets its own versioned directory

4. **Switch Between Projects**
   - Click "Recent Projects" in the sidebar
   - Select a project to load it
   - Or clear the project name to use temporary mode

### CLI

The CLI now supports project metadata parameters:

```bash
# Generate report with project metadata
markshark report results.csv \
  --out-xlsx exam_report.xlsx \
  --project-name "FINAL EXAM BIO101 2025" \
  --run-label "run_001_2026-01-21_1430"
```

## Handling Repeat Scoring

### What happens when you run scoring multiple times?

MarkShark **never overwrites** previous results. Instead, it creates a new versioned run folder:

**Scenario 1: Tweaking Parameters**
```
1. First run: results/run_001_2026-01-21_1430/
   - Score with default settings
2. Adjust bubble detection threshold
3. Second run: results/run_002_2026-01-21_1445/
   - Same students, different parameters
4. Both runs preserved for comparison
```

**Scenario 2: Second Batch of Tests**
```
1. First batch (20 students): results/run_001_2026-01-21_1000/
2. Second batch (15 students): results/run_002_2026-01-22_1400/
3. Both batches kept separate
4. Manually merge CSVs if needed, or generate separate reports
```

**Scenario 3: Redo After Error**
```
1. First attempt fails: results/run_001_2026-01-21_0900/
   - Contains partial/incorrect results
2. Fix issue and re-run: results/run_002_2026-01-21_0915/
   - Clean new results
3. Delete run_001 manually if desired
```

## Excel Report Enhancements

Reports now include project metadata on the Summary tab:

```
MarkShark Exam Report
─────────────────────────────────────
Project: FINAL EXAM BIO101 2025
Run: run_001_2026-01-21_1430
Generated: 2026-01-21 14:30:45
─────────────────────────────────────
Overall Exam Statistics
...
```

This helps you:
- Identify which exam a report belongs to (useful when opening files months later)
- Track when the report was generated
- Match reports to specific scoring runs

## Backward Compatibility

### Temporary Mode Still Works

If you **don't** enter a project name:
- MarkShark works exactly as before
- Files saved to temporary directories
- No project structure created
- Perfect for quick one-off tests

### Migration Path

You can gradually adopt Project Mode:
- Start using it for new exams
- Old temporary files remain unchanged
- No need to reorganize existing work

## Best Practices

### Naming Projects

**Good names:**
- `FINAL EXAM BIO101 2025`
- `Midterm Exam Physics Fall 2025`
- `Quiz 3 Chemistry Section A`

**Avoid:**
- Special characters: `/`, `\`, `:`, `*`, `?`, `"`, `<`, `>`, `|`
- Very long names (keep under 50 characters)
- Names that differ only by case

### Managing Runs

**Keep runs when:**
- Comparing different parameter settings
- Keeping separate batches of students
- Maintaining an audit trail

**Delete runs when:**
- A run clearly failed (corrupted output, wrong settings)
- Storage space is limited
- Only the final run is needed

**Merge runs when:**
- Multiple batches of the same exam
- Need a single consolidated report

Use spreadsheet software or pandas to merge CSVs:
```python
import pandas as pd

# Merge multiple runs
run1 = pd.read_csv("run_001/results.csv")
run2 = pd.read_csv("run_002/results.csv")
merged = pd.concat([run1, run2], ignore_index=True)
merged.to_csv("merged_results.csv", index=False)
```

## Troubleshooting

### "Project folder exists" but it's empty
- This happens if you created a project but haven't run any workflows yet
- Just run Quick Grade or Score to populate it

### Can't find my old temporary files
- Temporary files are in your working directory with prefixes like `score_`, `align_`
- They're not automatically migrated to project mode
- Consider using Project Mode for new work

### Runs numbered incorrectly
- MarkShark scans the `results/` folder for existing runs
- If you manually deleted run_002, the next run will be run_003 (not run_002)
- This ensures no accidental overwrites

### Project name contains weird characters
- MarkShark automatically sanitizes names for filesystem safety
- Spaces become underscores: `FINAL EXAM` → `FINAL_EXAM`
- Special characters are removed or replaced

## Implementation Details

### New Files

- **`src/markshark/project_utils.py`** - Project directory management utilities
  - `sanitize_project_name()` - Clean names for filesystem
  - `create_project_structure()` - Create directory tree
  - `create_run_directory()` - Version new runs
  - `get_next_run_number()` - Find next available run number
  - `find_projects()` - Discover existing projects

### Modified Files

- **`src/markshark/app_streamlit.py`** - GUI integration
  - Added "Current Project" field to sidebar
  - Updated all workflows (Quick Grade, Align, Score, Report, Viz)
  - Added "Recent Projects" browser

- **`src/markshark/tools/report_tools.py`** - Report metadata
  - Added `project_name` and `run_label` parameters
  - Updated Summary tab to show project info

- **`src/markshark/cli.py`** - CLI parameters
  - Added `--project-name` and `--run-label` to `report` command

## API Reference

### `project_utils.py`

```python
def sanitize_project_name(name: str) -> str:
    """Convert user input to filesystem-safe name."""

def create_project_structure(base_dir: Path, project_name: str) -> Path:
    """Create input/, aligned/, results/, config/ subdirectories."""

def create_run_directory(project_dir: Path, timestamp: Optional[datetime] = None) -> Tuple[Path, str]:
    """Create versioned run directory, returns (path, label)."""

def get_next_run_number(project_dir: Path) -> int:
    """Find next available run number."""

def get_project_info(project_dir: Path) -> dict:
    """Get metadata: name, num_runs, last_run, created."""

def find_projects(base_dir: Path) -> list[dict]:
    """Scan for all project directories."""
```

### CLI

```bash
markshark report INPUT_CSV \
  --out-xlsx REPORT.xlsx \
  --roster ROSTER.csv \
  --project-name "Project Name" \
  --run-label "run_001_2026-01-21_1430"
```

## Future Enhancements

Potential additions for future versions:

- **Project Settings** - Save default parameters per project
- **Run Comparison** - GUI tool to diff results between runs
- **Batch Merging** - Automatic merging of multiple runs
- **Project Export** - Zip entire project for archiving
- **Run Notes** - Attach notes/comments to specific runs
- **Template Locking** - Ensure same template used across runs

## Questions?

For issues, feature requests, or questions:
- GitHub: https://github.com/navarrew/markshark/issues
- Documentation: Check the main README.md

---

**Version:** 1.0.0
**Last Updated:** 2026-01-21
**Compatibility:** MarkShark v0.1.0+
