#!/usr/bin/env python3
"""
MarkShark
cli.py  —  MarkShark command line engine
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

# Config loader that supports a YAML (.yaml/.yml) formatted map of the bubble sheet
from .tools.bubblemap_io import load_bublmap
from .template_manager import TemplateManager, list_available_templates, get_template_by_name

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
from .visualize_core import overlay_bublmap
from .score_core import score_pdf
# stats_tools imported by report command when needed

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="MarkShark: align, visualize, score, and analyze bubble-sheet exams.",
)

# ---------------------- ALIGN ----------------------
@app.command()
def align(
    input_pdf: str = typer.Argument(..., help="Raw scans PDF"),
    template: str = typer.Option(..., "--template", "-t", help="Template PDF to align to"),
    out_pdf: str = typer.Option("aligned_scans.pdf", "--out-pdf", "-o", help="Output aligned PDF"),
    dpi: int = typer.Option(RENDER_DEFAULTS.dpi, "--dpi", help="Render DPI for alignment & output"),
    template_page: int = typer.Option(1, "--template-page", help="Template page index to use (1-based)"),
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
                rprint(f"[cyan]Alignment mode:[/cyan] fast (coarse-to-fine)")
            elif align_method == "fast":
                rprint(f"[cyan]Alignment mode:[/cyan] fast (coarse-to-fine)")
        except Exception as e:
            rprint(f"[yellow]Warning: Could not load bubblemap {bubblemap_path}: {e}[/yellow]")
            rprint("[yellow]Falling back to slow alignment mode.[/yellow]")
    else:
        if align_method == "fast":
            rprint(f"[yellow]Warning: 'fast' alignment requires --bubblemap. Using 'slow' mode.[/yellow]")
        elif align_method in ("auto", "slow"):
            rprint(f"[cyan]Alignment mode:[/cyan] slow (full-res ORB)")
    
    out = align_pdf_scans(
        input_pdf=input_pdf,
        template=template,
        out_pdf=out_pdf,
        dpi=dpi,
        template_page=template_page,
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


# --------------------------- VISUALIZE --------------------------
@app.command()
def visualize(
    input_pdf: str = typer.Argument(..., help="An aligned page PDF or template PDF"),
    bublmap: str = typer.Option(..., "--bublmap", "-m", help="Bubblemap file (.yaml/.yml)"),
    out_image: str = typer.Option("bubblemap_overlay.png", "--out-image", "-o", help="Output overlay image (png/jpg/pdf)"),
    pdf_renderer: str = typer.Option("auto", "--pdf-renderer", help="PDF renderer: auto|fitz|pdf2image"),
    dpi: int = typer.Option(RENDER_DEFAULTS.dpi, "--dpi", help="Render DPI"),
):
    """
    Overlay the bublmap bubble zones on top of a PDF page to verify placement.
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
        rprint(f"[red]Visualization failed for {bublmap}:[/red] {e}")
        raise typer.Exit(code=2)

    rprint(f"[green]Wrote:[/green] {out_image}")
    
    
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
    # NEW: Review/flagging options
    review_pdf: Optional[str] = typer.Option(
        None,
        "--review-pdf",
        help="Output PDF containing only pages with flagged answers (blank/multi). Use for manual review.",
    ),
    flagged_csv: Optional[str] = typer.Option(
        None,
        "--flagged-csv",
        help="Output CSV listing all flagged items (blank/multi answers) with student ID, question, and issue.",
    ),
    annotate_all_cells: bool = typer.Option(False, "--annotate-all-cells", help="Draw every bubble in each row"),
    label_density: bool = typer.Option(False, "--label-density", help="Overlay % fill text at bubble centers"),
    dpi: int = typer.Option(RENDER_DEFAULTS.dpi, "--dpi", help="Scan/PDF render DPI"),
    min_fill: Optional[float] = typer.Option(
        None,
        "--min-fill",
        help=f"""Minimum fraction of the darkest bubble required to consider a mark filled (default: {SCORING_DEFAULTS.min_fill}).
        Increase to require more completely filled bubbles; decrease to accept lighter or partially filled marks."""
    ),
    top2_ratio: Optional[float] = typer.Option(None, "--top2-ratio", help=f"default {SCORING_DEFAULTS.top2_ratio}"),
    min_score: Optional[float] = typer.Option(
        None,
        "--min-score",
        help=f"""Minimum score required to accept a bubble as filled (default: {SCORING_DEFAULTS.min_score}).
        Increase to require higher confidence in filled bubbles; decrease to accept lower scores."""
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
    # NEW: Inline stats option
    include_stats: bool = typer.Option(
        True,
        "--include-stats/--no-include-stats",
        help="Append basic statistics (mean, std, KR-20, item difficulty, point-biserial) to CSV. Requires answer key.",
    ),
):
    """
    Grade aligned scans using axis-based bublmap.
    
    When --key-txt is provided and --include-stats is enabled (default), the output CSV
    will include summary rows at the bottom with:
    - Exam statistics: N, Mean, StdDev, High/Low scores, KR-20 reliability
    - Item statistics: % correct and point-biserial for each question
    """
    try:
        _ = load_bublmap(bublmap)
    except Exception as e:
        rprint(f"[red]Failed to load bublmap {bublmap}:[/red] {e}")
        raise typer.Exit(code=2)

    try:
        scoring = apply_scoring_overrides(
            min_fill=min_fill if min_fill is not None else SCORING_DEFAULTS.min_fill,
            top2_ratio=top2_ratio if top2_ratio is not None else SCORING_DEFAULTS.top2_ratio,
            min_score=min_score if min_score is not None else SCORING_DEFAULTS.min_score,
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
            top2_ratio=scoring.top2_ratio,
            min_score=scoring.min_score,
            fixed_thresh=fixed_thresh if fixed_thresh is not None else scoring.fixed_thresh,
            auto_calibrate_thresh=scoring.auto_calibrate_thresh,
            verbose_calibration=scoring.verbose_calibration,
            annotate_all_cells=annotate_all_cells,
            label_density=label_density,
            review_pdf=review_pdf,
            flagged_csv=flagged_csv,
            include_stats=include_stats,
        )
    except Exception as e:
        rprint(f"[red]Scoring failed:[/red] {e}")
        raise typer.Exit(code=2)

    rprint(f"[green]Wrote:[/green] {out_csv}")
    if include_stats and key_txt:
        rprint(f"[cyan]Stats included:[/cyan] See bottom of {out_csv} for exam & item statistics")
    if review_pdf:
        rprint(f"[yellow]Review PDF:[/yellow] {review_pdf}")
    if flagged_csv:
        rprint(f"[yellow]Flagged CSV:[/yellow] {flagged_csv}")


# ---------------------- REPORT ----------------------
@app.command()
def report(
    input_csv: str = typer.Argument(..., help="Results CSV from 'score'"),
    out_xlsx: str = typer.Option("exam_report.xlsx", "--out-xlsx", "-o", help="Output Excel report"),
    roster_csv: Optional[str] = typer.Option(None, "--roster", "-r", help="Optional class roster CSV (StudentID, LastName, FirstName)"),
):
    """
    Generate an Excel report with per-version tabs, item analysis, and roster checking.

    The report includes:
    - Summary tab with overall exam statistics
    - Per-version tabs with student results and item statistics
    - Roster matching (if --roster provided): flags absent students and orphan scans
    - Color-coded item quality indicators
    """
    try:
        from .tools import report_tools
        report_tools.generate_report(
            input_csv=input_csv,
            out_xlsx=out_xlsx,
            roster_csv=roster_csv,
        )
        rprint(f"[green]Report generated:[/green] {out_xlsx}")
    except Exception as e:
        rprint(f"[red]Report generation failed:[/red] {e}")
        raise typer.Exit(code=2)


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
                    rprint(f"  [green]✓ Valid[/green]")
                else:
                    rprint(f"  [red]✗ Invalid:[/red]")
                    for error in errors:
                        rprint(f"    - {error}")
            rprint()
            
    except Exception as e:
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=2)


# ------------------------------- QUICK-GRADE -------------------------------
@app.command()
def quick_grade(
    input_pdf: str = typer.Argument(..., help="Raw student scans PDF"),
    template_id: str = typer.Option(..., "--template", "-t", help="Template ID or display name (use 'markshark templates' to list)"),
    key_txt: Optional[str] = typer.Option(None, "--key-txt", "-k", help="Answer key file (optional)"),
    out_csv: str = typer.Option("quick_grade_results.csv", "--out-csv", "-o", help="Output CSV of results"),
    out_pdf: str = typer.Option("quick_grade_annotated.pdf", "--out-pdf", help="Output annotated PDF"),
    out_dir: Optional[str] = typer.Option(None, "--out-dir", help="Output directory (default: same as out_csv)"),
    dpi: int = typer.Option(RENDER_DEFAULTS.dpi, "--dpi", help="Render DPI"),
    templates_dir: Optional[str] = typer.Option(None, "--templates-dir", help="Custom templates directory"),
    # Alignment options
    align_method: str = typer.Option("auto", "--align-method", help="Alignment method: auto|aruco|feature"),
    min_markers: int = typer.Option(ALIGN_DEFAULTS.min_aruco, "--min-markers", help="Min ArUco markers to accept"),
    # Scoring options
    min_fill: Optional[float] = typer.Option(None, "--min-fill", help=f"Min fill threshold (default: {SCORING_DEFAULTS.min_fill})"),
    top2_ratio: Optional[float] = typer.Option(None, "--top2-ratio", help=f"Top2 ratio (default: {SCORING_DEFAULTS.top2_ratio})"),
    min_score: Optional[float] = typer.Option(None, "--min-score", help=f"Min score (default: {SCORING_DEFAULTS.min_score})"),
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
            rprint(f"[cyan]Bubble grid alignment fallback:[/cyan] enabled")
        except Exception as e:
            rprint(f"[yellow]Warning: Could not load bubblemap: {e}[/yellow]")
            rprint("[yellow]Bubble grid alignment fallback will not be available.[/yellow]")
        
        # Determine output directory
        if out_dir is None:
            out_dir = str(Path(out_csv).parent) if Path(out_csv).parent != Path('.') else "."
        
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Align (with bubblemap for bubble grid fallback)
        rprint("[cyan]Step 1/2: Aligning scans...[/cyan]")
        aligned_pdf = out_dir_path / "aligned_scans.pdf"
        
        align_out = align_pdf_scans(
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
        
        scoring = apply_scoring_overrides(
            min_fill=min_fill if min_fill is not None else SCORING_DEFAULTS.min_fill,
            top2_ratio=top2_ratio if top2_ratio is not None else SCORING_DEFAULTS.top2_ratio,
            min_score=min_score if min_score is not None else SCORING_DEFAULTS.min_score,
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
            top2_ratio=scoring.top2_ratio,
            min_score=scoring.min_score,
            auto_calibrate_thresh=scoring.auto_calibrate_thresh,
            annotate_all_cells=annotate_all_cells,
            label_density=label_density,
        )
        
        rprint(f"[green]✅ Quick grade complete![/green]")
        rprint(f"[green]Results:[/green] {out_csv}")
        rprint(f"[green]Annotated PDF:[/green] {out_pdf}")
        rprint(f"[green]Aligned scans:[/green] {aligned_pdf}")
        
    except Exception as e:
        rprint(f"[red]Quick grade failed:[/red] {e}")
        raise typer.Exit(code=2)


# ------------------------------- GUI ---------------------------------
@app.command()
def gui(
    port: int = typer.Option(8501, "--port", help="Port to serve Streamlit GUI"),
    browser: bool = typer.Option(True, "--open-browser/--no-open-browser", help="Open browser automatically"),
):
    """
    Launch the Streamlit GUI.
    """
    # Resolve the path to the app module
    app_py = (Path(__file__).resolve().parent / "app_streamlit.py")
    if not app_py.exists():
        rprint(f"[red]Cannot locate app_streamlit.py at {app_py}[/red]")
        raise typer.Exit(code=2)

    cmd = ["streamlit", "run", str(app_py), "--server.port", str(port)]
    if not browser:
        cmd.extend(["--server.headless", "true"])

    rprint(f"[cyan]Launching:[/cyan] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        rprint("[red]Streamlit not found. Install it in your environment (`pip install streamlit`).[/red]")
        raise typer.Exit(code=3)
    except subprocess.CalledProcessError as e:
        rprint(f"[red]Streamlit exited with error:[/red] {e}")
        raise typer.Exit(code=4)


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
