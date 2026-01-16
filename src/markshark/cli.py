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
from .tools import stats_tools as stats_mod  # has run(...)

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
    align_method: str = typer.Option("auto", "--align-method", help="Alignment pipeline: auto|aruco|feature"),
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
):
    """
    Align raw scans to a template PDF.
    """
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
    ),    # (annotation flags you already added can stay)
):
    """
    Grade aligned scans using axis-based bublmap.
    """
    try:
        _ = load_bublmap(bublmap)
    except Exception as e:
        rprint(f"[red]Failed to load bublmap {bublmap}:[/red] {e}")
        raise typer.Exit(code=2)

    try:
        scoring = apply_scoring_overrides(  # ← use centralized helper in defaults.py
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
            fixed_thresh=fixed_thresh,
            auto_calibrate_thresh=scoring.auto_calibrate_thresh,
            verbose_calibration=scoring.verbose_calibration,
            annotate_all_cells=annotate_all_cells,
            label_density=label_density,
        )
    except Exception as e:
        rprint(f"[red]Grading failed:[/red] {e}")
        raise typer.Exit(code=2)

    rprint(f"[green]Wrote results:[/green] {out_csv}")


# ------------------------------ STATS -------------------------------
@app.command()
def stats(
    input_csv: str = typer.Argument(..., help="Results CSV (from 'score')"),
    output_csv: str = typer.Option("results_with_stats.csv", "--output-csv", "-o", help="Augmented CSV with summary rows"),
    item_pattern: str = typer.Option(r"^Q\d+$", "--item-pattern", help="Regex for item columns (default: ^Q\\d+$). Example: '^Q\\d+$'"),
    percent: bool = typer.Option(True, "--percent/--proportion", help="Report difficulty as percent (0-100) (default) or proportion (0-1)"),
    label_col: Optional[str] = typer.Option("name", "--label-col", help="Column containing student label (name/id)"),
    exam_stats_csv: Optional[str] = typer.Option(None, "--exam-stats-csv", help="Optional CSV with KR-20/KR-21, mean, SD"),
    plots_dir: Optional[str] = typer.Option(None, "--plots-dir", help="Optional directory for IRT-ish item plots"),
    key_row_index: Optional[int] = typer.Option(None, "--key-row-index", help="Row index of answer key (0-based). Auto-detect if omitted."),
    answers_mode: str = typer.Option("letters", "--answers-mode", help="letters|index depending on how answers are stored"),
    item_report_csv: Optional[str] = typer.Option(None, "--item-report-csv", help="Optional per-item distractor report CSV"),
    key_label: str = typer.Option("KEY", "--key-label", help="Label string for the key row used in auto-detection"),
    decimals: int = typer.Option(3, "--decimals", help="Number of decimals for output rounding (default: 3)"),
):
    """
    Compute item difficulty, point-biserial, and exam reliability (KR-20/KR-21).
    """
    try:
        # Newer stats_tools.run signatures accept 'decimals'; older ones don't.
        stats_mod.run(
            input_csv=input_csv,
            output_csv=output_csv,
            item_pattern=item_pattern,
            percent=percent,
            label_col=label_col,
            exam_stats_csv=exam_stats_csv,
            plots_dir=plots_dir,
            key_row_index=key_row_index,
            answers_mode=answers_mode,
            item_report_csv=item_report_csv,
            key_label=key_label,
            decimals=decimals,  # default 3
        )
    except TypeError:
        # Backward-compat: call without 'decimals'
        stats_mod.run(
            input_csv=input_csv,
            output_csv=output_csv,
            item_pattern=item_pattern,
            percent=percent,
            label_col=label_col,
            exam_stats_csv=exam_stats_csv,
            plots_dir=plots_dir,
            key_row_index=key_row_index,
            answers_mode=answers_mode,
            item_report_csv=item_report_csv,
            key_label=key_label,
        )
        rprint("[yellow]Note:[/yellow] your stats_tools.run() doesn’t support a 'decimals' parameter; "
               "update it to get consistent 3-decimal rounding in all outputs.")
    rprint(f"[green]Wrote stats:[/green] {output_csv}")
    if exam_stats_csv:
        rprint(f"[green]Exam summary:[/green] {exam_stats_csv}")
    if item_report_csv:
        rprint(f"[green]Item report:[/green] {item_report_csv}")



# ------------------------------- TEMPLATES -------------------------------
@app.command()
def templates(
    templates_dir: Optional[str] = typer.Option(None, "--templates-dir", "-d", help="Custom templates directory (default: auto-detect)"),
    create_example: bool = typer.Option(False, "--create-example", help="Create an example template structure"),
    validate: bool = typer.Option(False, "--validate", help="Validate all templates"),
):
    """
    List available bubble sheet templates.
    """
    try:
        manager = TemplateManager(templates_dir)
        rprint(f"[cyan]Templates directory:[/cyan] {manager.templates_dir}")
        rprint()
        
        if create_example:
            example_dir = manager.create_example_template()
            rprint(f"[green]Created example template at:[/green] {example_dir}")
            rprint("[yellow]Note:[/yellow] You'll need to add actual PDF and populate the YAML with bubble coordinates.")
            return
        
        templates = manager.scan_templates()
        
        if not templates:
            rprint("[yellow]No templates found.[/yellow]")
            rprint("Create a template directory with:")
            rprint("  - templates/your_template_name/master_template.pdf")
            rprint("  - templates/your_template_name/bubblemap.yaml")
            return
        
        rprint(f"[cyan]Found {len(templates)} template(s):[/cyan]")
        rprint()
        
        for template in templates:
            rprint(f"[bold]{template.display_name}[/bold]")
            rprint(f"  ID: {template.template_id}")
            if template.description:
                rprint(f"  Description: {template.description}")
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
        
        # Determine output directory
        if out_dir is None:
            out_dir = str(Path(out_csv).parent) if Path(out_csv).parent != Path('.') else "."
        
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Align
        rprint("[cyan]Step 1/2: Aligning scans...[/cyan]")
        aligned_pdf = out_dir_path / "aligned_scans.pdf"
        
        align_out = align_pdf_scans(
            input_pdf=input_pdf,
            template=str(template.template_pdf_path),
            out_pdf=str(aligned_pdf),
            dpi=dpi,
            align_method=align_method,
            min_markers=min_markers,
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

