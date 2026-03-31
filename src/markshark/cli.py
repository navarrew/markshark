#!/usr/bin/env python3
"""
MarkShark
cli.py  —  MarkShark command line engine
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

# Config loader that supports a YAML (.yaml/.yml) formatted map of the bubble sheet
from .tools.bubblemap_io import load_bublmap
from .template_manager import TemplateManager, get_template_by_name

from .defaults import (
    SCORING_DEFAULTS,
    FEAT_DEFAULTS,
    MATCH_DEFAULTS,
    EST_DEFAULTS,
    ALIGN_DEFAULTS,
    RENDER_DEFAULTS,
    apply_scoring_overrides,
)
# Core modules
from .align_core import align_pdf_scans
from .mapviewer_core import overlay_bublmap
from .score_core import score_pdf
# stats_tools imported by report command when needed

def _version_callback(value: bool):
    """Print version and exit when --version is passed."""
    if value:
        from markshark import __version__
        rprint(f"MarkShark {__version__}")
        raise typer.Exit()


app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
):
    """MarkShark: align, mapviewer, score, and analyze bubble-sheet exams."""

# ------------------------------- QUICK-GRADE -------------------------------
@app.command()
def quick_grade(
    input_pdf: str = typer.Argument(..., help="Raw student scans PDF"),
    template_id: str = typer.Option(..., "--template", "-t", help="Template ID or display name (use 'markshark templates' to list)"),
    key_txt: Optional[str] = typer.Option(None, "--key-txt", "-k", help="Answer key file (optional)"),
    out_csv: str = typer.Option("score_data/results.csv", "--out-csv", "-o", help="Output CSV of results"),
    out_pdf: str = typer.Option("scored_scans.pdf", "--out-pdf", help="Output annotated PDF"),
    out_dir: Optional[str] = typer.Option(None, "--out-dir", help="Output directory (project root; default: same as out_csv parent)"),
    dpi: int = typer.Option(RENDER_DEFAULTS.dpi, "--dpi", help="Render DPI"),
    templates_dir: Optional[str] = typer.Option(None, "--templates-dir", help="Custom templates directory"),
    # Alignment options
    align_method: str = typer.Option("auto", "--align-method", help="Alignment method: auto|aruco|feature"),
    min_markers: int = typer.Option(ALIGN_DEFAULTS.min_aruco, "--min-markers", help="Min ArUco markers to accept"),
    # Scoring options (user provides integers 0-100, we convert to fractions internally)
    min_fill: Optional[int] = typer.Option(None, "--min-fill", help=f"Min fill score (0-100) to accept as filled (default: {SCORING_DEFAULTS.min_fill}). Matches the scores shown on annotated PDFs."),
    annotate_all_cells: bool = typer.Option(False, "--annotate-all-cells", help="Draw every bubble in each row"),
    label_density: bool = typer.Option(False, "--label-density", help="Overlay % fill text"),
    auto_thresh: bool = typer.Option(SCORING_DEFAULTS.auto_calibrate_thresh, "--auto-thresh/--no-auto-thresh", help="Auto-calibrate threshold"),
):
    """
    Quick grade: align + score in one command using a template.
    
    This command automatically uses bubble grid alignment as a fallback when
    ArUco markers are not detected, using the bubble positions from the template's
    bubblemap YAML.
    """
    try:
        # Get template
        template = get_template_by_name(template_id, templates_dir)
        if not template:
            rprint(f"[red]Template not found:[/red] {template_id}")
            rprint("[yellow]Available templates:[/yellow]")
            manager = TemplateManager(templates_dir)
            for t in manager.scan_templates():
                rprint(f"  - {t.display_name} (ID: {t.template_id})")
            raise typer.Exit(code=2)
        
        rprint(f"[cyan]Using template:[/cyan] {template.display_name}")
        
        # Load bubblemap for bubble grid alignment fallback
        bubblemap = None
        try:
            bubblemap = load_bublmap(str(template.bubblemap_yaml_path))
            rprint("[cyan]Bubble grid alignment fallback:[/cyan] enabled")
        except Exception as e:
            rprint(f"[yellow]Warning: Could not load bubblemap: {e}[/yellow]")
            rprint("[yellow]Bubble grid alignment fallback will not be available.[/yellow]")
        
        # Determine output directory
        if out_dir is None:
            out_dir = str(Path(out_csv).parent) if Path(out_csv).parent != Path('.') else "."
        
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        # Ensure parent directories for output files exist
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Align (with bubblemap for bubble grid fallback)
        rprint("[cyan]Step 1/2: Aligning scans...[/cyan]")
        input_files_dir = out_dir_path / "input_files"
        input_files_dir.mkdir(parents=True, exist_ok=True)
        aligned_pdf = input_files_dir / "aligned_scans.pdf"
        
        align_pdf_scans(
            input_pdf=input_pdf,
            template=str(template.template_pdf_path),
            out_pdf=str(aligned_pdf),
            dpi=dpi,
            align_method=align_method,
            min_markers=min_markers,
            bubblemap=bubblemap,  # NEW: Pass bubblemap for bubble grid fallback
        )
        rprint(f"[green]✓ Alignment complete:[/green] {aligned_pdf}")
        
        # Step 2: Score
        rprint("[cyan]Step 2/2: Scoring sheets...[/cyan]")
        
        # Convert user-facing integers (0-100) to internal fractions (0-1)
        min_fill_int = min_fill if min_fill is not None else SCORING_DEFAULTS.min_fill
        min_fill_frac = min_fill_int / 100.0

        scoring = apply_scoring_overrides(
            min_fill=min_fill_frac,
            auto_calibrate_thresh=auto_thresh,
        )

        score_pdf(
            input_path=str(aligned_pdf),
            bublmap_path=str(template.bubblemap_yaml_path),
            out_csv=out_csv,
            key_txt=key_txt,
            out_pdf=out_pdf,
            dpi=dpi,
            min_fill=scoring.min_fill,
            auto_calibrate_thresh=scoring.auto_calibrate_thresh,
            calibrate_background=scoring.calibrate_background,
            background_percentile=scoring.background_percentile,
            adaptive_rescoring=scoring.adaptive_rescoring,
            adaptive_max_adjustment=scoring.adaptive_max_adjustment,
            adaptive_min_above_floor=scoring.adaptive_min_above_floor,
            annotate_all_cells=annotate_all_cells,
            label_density=label_density,
        )
        
        rprint("[green]✅ Quick grade complete![/green]")
        rprint(f"[green]Results:[/green] {out_csv}")
        rprint(f"[green]Annotated PDF:[/green] {out_pdf}")
        rprint(f"[green]Aligned scans:[/green] {aligned_pdf}")
        
    except Exception as e:
        rprint(f"[red]Quick grade failed:[/red] {e}")
        raise typer.Exit(code=2)



# ---------------------- ALIGN ----------------------
@app.command()
def align(
    input_pdf: str = typer.Argument(..., help="Raw scans PDF"),
    template: str = typer.Option(..., "--template", "-t", help="Template PDF to align to"),
    out_pdf: str = typer.Option("aligned_scans.pdf", "--out-pdf", "-o", help="Output aligned PDF"),
    dpi: int = typer.Option(RENDER_DEFAULTS.dpi, "--dpi", help="Render DPI for alignment & output"),
    align_method: str = typer.Option(
        "auto", "--align-method",
        help="Alignment method: auto|fast|slow|aruco. "
             "fast=coarse-to-fine (72 DPI ORB + bubble grid, requires --bubblemap), "
             "slow=full-res ORB only, "
             "auto=fast if bubblemap provided else slow"
    ),
    estimator_method: str = typer.Option(EST_DEFAULTS.estimator_method, "--estimator-method", help="Homography estimator: auto|ransac|usac"),
    min_markers: int = typer.Option(ALIGN_DEFAULTS.min_aruco, "--min-markers", help="Min ArUco markers to accept"),
    ransac: float = typer.Option(EST_DEFAULTS.ransac_thresh, "--ransac", help="RANSAC reprojection threshold"),
    use_ecc: bool = typer.Option(EST_DEFAULTS.use_ecc, "--use-ecc/--no-use-ecc", help="Enable ECC refinement"),
    ecc_max_iters: int = typer.Option(EST_DEFAULTS.ecc_max_iters, "--ecc-max-iters", help="ECC iterations"),
    ecc_eps: float = typer.Option(EST_DEFAULTS.ecc_eps, "--ecc-eps", help="ECC termination epsilon"),
    orb_nfeatures: int = typer.Option(FEAT_DEFAULTS.orb_nfeatures, "--orb-nfeatures", help="ORB features for feature-based align"),
    match_ratio: float = typer.Option(MATCH_DEFAULTS.ratio_test, "--match-ratio", help="Lowe's ratio test for feature matching"),
    dict_name: str = typer.Option(ALIGN_DEFAULTS.dict_name, "--dict-name", help="ArUco dictionary"),
    first_page: Optional[int] = typer.Option(None, "--first-page", help="First page index (1-based)"),
    last_page: Optional[int] = typer.Option(None, "--last-page", help="Last page index (inclusive, 1-based)"),
    # NEW: Bubblemap for bubble grid alignment fallback
    bubblemap_path: Optional[str] = typer.Option(
        None, "--bubblemap", "-m",
        help="Bubblemap YAML file. Enables 'fast' alignment mode (coarse-to-fine with bubble grid)."
    ),
):
    """
    Align raw scans to a template PDF.
    
    ALIGNMENT METHODS:
    
    - auto: Uses 'fast' if --bubblemap provided, else 'slow' (recommended)
    - fast: Coarse-to-fine alignment. Quick 72 DPI ORB pass, then bubble grid
            refinement at full res. Requires --bubblemap. Best for bubble sheets.
    - slow: Full resolution ORB alignment. More thorough but slower.
            Works without bubblemap.
    - aruco: ArUco marker alignment only. Requires markers on the sheet.
    """
    # Load bubblemap if provided (for bubble grid fallback)
    bubblemap = None
    if bubblemap_path:
        try:
            bubblemap = load_bublmap(bubblemap_path)
            rprint(f"[cyan]Loaded bubblemap:[/cyan] {bubblemap_path}")
            if align_method == "auto":
                rprint("[cyan]Alignment mode:[/cyan] fast (coarse-to-fine)")
            elif align_method == "fast":
                rprint("[cyan]Alignment mode:[/cyan] fast (coarse-to-fine)")
        except Exception as e:
            rprint(f"[yellow]Warning: Could not load bubblemap {bubblemap_path}: {e}[/yellow]")
            rprint("[yellow]Falling back to slow alignment mode.[/yellow]")
    else:
        if align_method == "fast":
            rprint("[yellow]Warning: 'fast' alignment requires --bubblemap. Using 'slow' mode.[/yellow]")
        elif align_method in ("auto", "slow"):
            rprint("[cyan]Alignment mode:[/cyan] slow (full-res ORB)")
    
    out = align_pdf_scans(
        input_pdf=input_pdf,
        template=template,
        out_pdf=out_pdf,
        dpi=dpi,
        align_method=align_method,
        estimator_method=estimator_method,
        dict_name=dict_name,
        min_markers=min_markers,
        ransac=ransac,
        use_ecc=use_ecc,
        ecc_max_iters=ecc_max_iters,
        ecc_eps=ecc_eps,
        orb_nfeatures=orb_nfeatures,
        match_ratio=match_ratio,
        first_page=first_page,
        last_page=last_page,
        bubblemap=bubblemap,  # NEW: Pass bubblemap for bubble grid fallback
    )
    rprint(f"[green]Wrote:[/green] {out}")


# ---------------------- SCORE ----------------------
@app.command()
def score(
    input_pdf: str = typer.Argument(..., help="Aligned scans PDF"),
    bublmap: str = typer.Option(..., "--bublmap", "-c", help="Bubblemap file (.yaml/.yml)"),
    key_txt: Optional[str] = typer.Option(None, "--key-txt", "-k",
        help="Answer key file (A/B/C/... one per line). If provided, only first len(key) questions are scored."),
    out_csv: str = typer.Option("results.csv", "--out-csv", "-o", help="Output CSV of per-student results"),
    out_annotated_dir: Optional[str] = typer.Option(None, "--out-annotated-dir", help="Directory to write annotated sheets"),
    out_pdf: Optional[str] = typer.Option(
        None,
        "--out-pdf",
        help=f"Annotated PDF output filename. Default: {SCORING_DEFAULTS.out_pdf}. Use \"\"\" to disable.",
    ),
    roster_csv: Optional[str] = typer.Option(
        None,
        "--roster-csv",
        help="Class roster CSV with StudentID column. Flags orphan scans (ID not in roster) and tracks absent students.",
    ),
    annotate_all_cells: bool = typer.Option(False, "--annotate-all-cells", help="Draw every bubble in each row"),
    label_density: bool = typer.Option(False, "--label-density", help="Overlay % fill text at bubble centers"),
    dpi: int = typer.Option(RENDER_DEFAULTS.dpi, "--dpi", help="Scan/PDF render DPI"),
    min_fill: Optional[int] = typer.Option(
        None,
        "--min-fill",
        help=f"""Minimum fill score (0-100) to consider a bubble filled (default: {SCORING_DEFAULTS.min_fill}).
        This matches the scores shown on annotated PDFs. Increase to require darker marks; decrease to accept lighter marks."""
    ),
    fixed_thresh: Optional[int] = typer.Option(None, "--fixed-thresh", help=f"default {SCORING_DEFAULTS.fixed_thresh}"),
    auto_thresh: bool = typer.Option(
        SCORING_DEFAULTS.auto_calibrate_thresh,
        "--auto-thresh/--no-auto-thresh",
        help="Auto tune fixed_thresh per page when --fixed-thresh is omitted",
    ),
    verbose_calibration: bool = typer.Option(
        SCORING_DEFAULTS.verbose_calibration,
        "--verbose-thresh",
        help="Print per-page threshold calibration diagnostics",
    ),
):
    """
    Grade aligned scans using axis-based bublmap.

    Outputs a simplified CSV with:
    - Header row with Page, Version, LastName, FirstName, StudentID, Correct, Incorrect, Blank, Multi, Flagged, FlagDetails, Q1...
    - KEY row(s) for each version
    - Student rows with scores and answers
    - Flagged column: "Y" if student has any flagged issues
    - FlagDetails column: pipe-separated flags like "Q5:blank|Q10:multi|ID:orphan"

    Use the 'report' command to generate statistics and Excel reports from this CSV.
    """
    try:
        _ = load_bublmap(bublmap)
    except Exception as e:
        rprint(f"[red]Failed to load bublmap {bublmap}:[/red] {e}")
        raise typer.Exit(code=2)

    try:
        # Convert user-facing integers (0-100) to internal fractions (0-1)
        min_fill_int = min_fill if min_fill is not None else SCORING_DEFAULTS.min_fill
        min_fill_frac = min_fill_int / 100.0

        scoring = apply_scoring_overrides(
            min_fill=min_fill_frac,
            fixed_thresh=SCORING_DEFAULTS.fixed_thresh,
            auto_calibrate_thresh=auto_thresh,
            verbose_calibration=verbose_calibration,
        )

        score_pdf(
            input_path=input_pdf,
            bublmap_path=bublmap,
            out_csv=out_csv,
            key_txt=key_txt,
            out_annotated_dir=out_annotated_dir,
            out_pdf=out_pdf,
            dpi=dpi,
            min_fill=scoring.min_fill,
            fixed_thresh=fixed_thresh if fixed_thresh is not None else scoring.fixed_thresh,
            auto_calibrate_thresh=scoring.auto_calibrate_thresh,
            verbose_calibration=scoring.verbose_calibration,
            calibrate_background=scoring.calibrate_background,
            background_percentile=scoring.background_percentile,
            adaptive_rescoring=scoring.adaptive_rescoring,
            adaptive_max_adjustment=scoring.adaptive_max_adjustment,
            adaptive_min_above_floor=scoring.adaptive_min_above_floor,
            annotate_all_cells=annotate_all_cells,
            label_density=label_density,
            roster_csv=roster_csv,
        )
    except Exception as e:
        rprint(f"[red]Scoring failed:[/red] {e}")
        raise typer.Exit(code=2)

    rprint(f"[green]Wrote:[/green] {out_csv}")
    rprint(f"[cyan]Format:[/cyan] Simplified CSV with Flagged/FlagDetails columns. Use 'report' command for statistics.")


# ---------------------- REPORT ----------------------
@app.command()
def report(
    input_csv: str = typer.Argument(..., help="Results CSV from 'score'"),
    out_xlsx: str = typer.Option("exam_report.xlsx", "--out-xlsx", "-o", help="Output Excel report"),
    roster_csv: Optional[str] = typer.Option(None, "--roster", "-r", help="Optional class roster CSV (StudentID, LastName, FirstName)"),
    project_name: Optional[str] = typer.Option(None, "--project-name", help="Assessment name to include in report header"),
    course_name: Optional[str] = typer.Option(None, "--course-name", help="Course name to include in report header"),
    run_label: Optional[str] = typer.Option(None, "--run-label", help="Run label (e.g., 2025-01-21_final) to include in report header"),
    corrections: Optional[str] = typer.Option(None, "--corrections", "--corrections-xlsx", help="Corrections file (.csv or .xlsx) to apply and list on Summary tab"),
    simple: bool = typer.Option(False, "--simple", help="Simple Grade report: class scores and answer key only (no item analysis)"),
):
    """
    Generate an Excel report with per-version tabs, item analysis, and roster checking.

    The report includes:
    - Summary tab with overall exam statistics
    - Per-version tabs with student results and item statistics
    - Roster matching (if --roster provided): flags absent students and orphan scans
    - Color-coded item quality indicators
    - Project metadata (if --project-name or --run-label provided)

    Use --simple for a streamlined report (class scores + answer key only).
    """
    try:
        from .tools import report_tools
        report_tools.generate_report(
            input_csv=input_csv,
            out_xlsx=out_xlsx,
            roster_csv=roster_csv,
            project_name=project_name,
            course_name=course_name,
            run_label=run_label,
            corrections_xlsx=corrections,
            simple=simple,
        )
        rprint(f"[green]Report generated:[/green] {out_xlsx}")
    except Exception as e:
        rprint(f"[red]Report generation failed:[/red] {e}")
        raise typer.Exit(code=2)

# --------------------------- MAPVIEWER --------------------------
@app.command()
def mapviewer(
    input_pdf: str = typer.Argument(..., help="An aligned page PDF or template PDF"),
    bublmap: str = typer.Option(..., "--bublmap", "-m", help="Bubblemap file (.yaml/.yml)"),
    out_image: str = typer.Option("bubblemap_overlay.png", "--out-image", "-o", help="Output overlay image (png/jpg/pdf)"),
    pdf_renderer: str = typer.Option("auto", "--pdf-renderer", help="PDF renderer: auto|fitz|pdf2image"),
    dpi: int = typer.Option(RENDER_DEFAULTS.dpi, "--dpi", help="Render DPI"),
):
    """
    Overlay the bublmap bubble zones on top of a PDF page to verify placement.

    Use this to visualize where MarkShark expects to find bubbles on your template.
    """
    try:
        overlay_bublmap(
            bublmap_path=bublmap,
            input_path=input_pdf,
            out_image=out_image,
            dpi=dpi,
            pdf_renderer=pdf_renderer,
        )
    except Exception as e:
        rprint(f"[red]Map viewer failed for {bublmap}:[/red] {e}")
        raise typer.Exit(code=2)

    rprint(f"[green]Wrote:[/green] {out_image}")
    
    
# ---------------------- TEMPLATES ----------------------
@app.command()
def templates(
    templates_dir: Optional[str] = typer.Option(None, "--templates-dir", "-d", help="Templates directory (default: auto-detect)"),
    validate: bool = typer.Option(False, "--validate", "-v", help="Validate each template"),
):
    """
    List available bubble sheet templates.
    """
    try:
        manager = TemplateManager(templates_dir)
        templates_list = manager.scan_templates(force_refresh=True)
        
        if not templates_list:
            rprint("[yellow]No templates found.[/yellow]")
            rprint(f"Templates directory: {manager.templates_dir}")
            return
        
        rprint(f"[cyan]Found {len(templates_list)} template(s) in {manager.templates_dir}:[/cyan]\n")
        
        for template in templates_list:
            rprint(f"[bold]{template.display_name}[/bold] (ID: {template.template_id})")
            if template.description:
                rprint(f"  {template.description}")
            if template.num_questions:
                rprint(f"  Questions: {template.num_questions}")
            if template.num_choices:
                rprint(f"  Choices: {template.num_choices}")
            rprint(f"  PDF: {template.template_pdf_path}")
            rprint(f"  YAML: {template.bubblemap_yaml_path}")
            
            if validate:
                is_valid, errors = manager.validate_template(template)
                if is_valid:
                    rprint("  [green]✓ Valid[/green]")
                else:
                    rprint("  [red]✗ Invalid:[/red]")
                    for error in errors:
                        rprint(f"    - {error}")
            rprint()
            
    except Exception as e:
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=2)


# ------------------------------- MOCK-DATASET ---------------------------------
@app.command()
def mock_dataset(
    template_id: str = typer.Option(..., "--template", "-t", help="Template ID or display name (use 'markshark templates' to list)"),
    out_dir: str = typer.Option(..., "--out-dir", "-o", help="Output directory for generated files"),
    num_students: int = typer.Option(100, "--num-students", "-n", help="Number of fake students to generate"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    dpi: int = typer.Option(150, "--dpi", help="DPI for rendered images"),
    templates_dir: Optional[str] = typer.Option(None, "--templates-dir", help="Custom templates directory"),
    darkness_min: float = typer.Option(0.4, "--darkness-min", help="Minimum bubble darkness (0-1)"),
    darkness_max: float = typer.Option(1.0, "--darkness-max", help="Maximum bubble darkness (0-1)"),
    apply_transform: bool = typer.Option(False, "--apply-transform", help="Apply random rotation/translation"),
    blank_rate: float = typer.Option(0.02, "--blank-rate", help="Rate of blank answers"),
    multi_rate: float = typer.Option(0.02, "--multi-rate", help="Rate of multi-fill answers"),
    num_id_errors: int = typer.Option(2, "--num-id-errors", help="Number of students with corrupted IDs"),
    num_missing_version: int = typer.Option(2, "--num-missing-version", help="Number of students with blank version field"),
    num_and_keys: int = typer.Option(0, "--num-and-keys", help="Number of AND key questions (e.g. B&C)"),
    num_or_keys: int = typer.Option(0, "--num-or-keys", help="Number of OR key questions (e.g. B^C)"),
    default_points: int = typer.Option(1, "--default-points", help="Default points per question"),
    num_double_points: int = typer.Option(0, "--num-double-points", help="Number of questions worth double points"),
):
    """
    Generate a mock dataset from a template for testing.

    Creates synthetic student scans with filled bubbles, an answer key,
    and a CSV with expected student responses.
    """
    try:
        # Get template
        template = get_template_by_name(template_id, templates_dir)
        if not template:
            rprint(f"[red]Template not found:[/red] {template_id}")
            rprint("[yellow]Available templates:[/yellow]")
            manager = TemplateManager(templates_dir)
            for t in manager.scan_templates():
                rprint(f"  - {t.display_name} (ID: {t.template_id})")
            raise typer.Exit(code=2)

        rprint(f"[cyan]Using template:[/cyan] {template.display_name}")

        from .mock_dataset import generate_mock_dataset

        results = generate_mock_dataset(
            template_path=str(template.template_pdf_path),
            bubblemap_path=str(template.bubblemap_yaml_path),
            out_dir=out_dir,
            num_students=num_students,
            seed=seed,
            dpi=dpi,
            darkness_min=darkness_min,
            darkness_max=darkness_max,
            apply_transform=apply_transform,
            blank_rate=blank_rate,
            multi_rate=multi_rate,
            num_id_errors=num_id_errors,
            num_missing_version=num_missing_version,
            num_and_keys=num_and_keys,
            num_or_keys=num_or_keys,
            default_points=default_points,
            num_double_points=num_double_points,
            verbose=True,
        )

        rprint(f"\n[green]Mock dataset generated![/green]")
        rprint(f"  Answer key: {results['answer_key']}")
        rprint(f"  Scans PDF: {results['scans']}")
        rprint(f"  Student responses CSV: {results['responses']}")

    except Exception as e:
        rprint(f"[red]Mock dataset generation failed:[/red] {e}")
        raise typer.Exit(code=2)


# ------------------------------- MAIN --------------------------------
def app_main(
    # Annotation styling overrides (B,G,R CSV)
    color_correct: Optional[str] = typer.Option(None, "--color-correct", help="BGR CSV for correct (e.g., 0,200,0)"),
    color_incorrect: Optional[str] = typer.Option(None, "--color-incorrect", help="BGR CSV for incorrect (e.g., 0,0,255)"),
    color_blank: Optional[str] = typer.Option(None, "--color-blank", help="BGR CSV for blank (e.g., 160,160,160)"),
    color_multi: Optional[str] = typer.Option(None, "--color-multi", help="BGR CSV for multi (e.g., 0,140,255)"),
    percent_text_color: Optional[str] = typer.Option(None, "--percent-text-color", help="BGR CSV for % labels"),
    color_zone: Optional[str] = typer.Option(None, "--color-zone", help="BGR CSV for name/ID zone circles"),
    thickness_answers: Optional[int] = typer.Option(None, "--thickness-answers", help="Circle thickness for answers"),
    thickness_names: Optional[int] = typer.Option(None, "--thickness-names", help="Circle thickness for names/IDs"),
    label_font_scale: Optional[float] = typer.Option(None, "--label-font-scale", help="Font scale for % labels"),
    label_thickness: Optional[int] = typer.Option(None, "--label-thickness", help="Font thickness for % labels")
) -> None:
    """Entry point for console_scripts."""
    try:
        app()
    except KeyboardInterrupt:
        rprint("\n[red]Interrupted[/red]")
        sys.exit(130)

if __name__ == "__main__":
    app_main()
