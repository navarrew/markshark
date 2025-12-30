from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

# Config loader that supports a YAML (.yaml/.yml) formatted map of the bubble sheet
from .config_io import load_config

from .defaults import (
    SCORING_DEFAULTS, FEAT_DEFAULTS, MATCH_DEFAULTS, EST_DEFAULTS, ALIGN_DEFAULTS, RENDER_DEFAULTS,
    ANNOTATION_DEFAULTS, apply_annotation_overrides, AnnotationDefaults,
    apply_scoring_overrides,
)

# Core modules
from .defaults import SCORING_DEFAULTS as DEFAULTS
from .align_core import align_pdf_scans
from .visualize_core import overlay_config
from .score_core import grade_pdf
from .tools import stats_tools as stats_mod  # has run(...)

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="MarkShark: align, visualize, grade, and analyze bubble-sheet exams.",
)

# ---------------------- ALIGN ----------------------
@app.command()
def align(
    input_pdf: str = typer.Argument(..., help="Raw scans PDF"),
    template: str = typer.Option(..., "--template", "-t", help="Template PDF to align to"),
    out_pdf: str = typer.Option("aligned_scans.pdf", "--out-pdf", "-o", help="Output aligned PDF"),
    dpi: int = typer.Option(RENDER_DEFAULTS.dpi, "--dpi", help="Render DPI for alignment & output"),
    template_page: int = typer.Option(1, "--template-page", help="Template page index to use (1-based)"),
    align_method: str = typer.Option("auto", "--align-method", "--align_method", help="Alignment pipeline: auto|aruco|feature"),
    estimator_method: str = typer.Option(EST_DEFAULTS.estimator_method, "--estimator-method", "--estimator_method", help="Homography estimator: auto|ransac|usac"),
    min_markers: int = typer.Option(ALIGN_DEFAULTS.min_aruco, "--min-markers", help="Min ArUco markers to accept"),
    ransac: float = typer.Option(EST_DEFAULTS.ransac_thresh, "--ransac", help="RANSAC reprojection threshold"),
    use_ecc: bool = typer.Option(EST_DEFAULTS.use_ecc, "--use-ecc/--no-use-ecc", help="Enable ECC refinement"),
    ecc_max_iters: int = typer.Option(EST_DEFAULTS.ecc_max_iters, "--ecc-max-iters", help="ECC iterations"),
    ecc_eps: float = typer.Option(EST_DEFAULTS.ecc_eps, "--ecc-eps", help="ECC termination epsilon"),
    orb_nfeatures: int = typer.Option(FEAT_DEFAULTS.orb_nfeatures, "--orb-nfeatures", help="ORB features for feature-based align"),
    match_ratio: float = typer.Option(MATCH_DEFAULTS.ratio_test, "--match-ratio", help="Lowe ratio for feature matching"),
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
    config: str = typer.Option(..., "--config", "-c", help="Config file (.yaml/.yml)"),
    out_image: str = typer.Option("config_overlay.png", "--out-image", "-o", help="Output overlay image (png/jpg/pdf)"),
    pdf_renderer: str = typer.Option("auto", "--pdf-renderer", help="PDF renderer: auto|fitz|pdf2image"),
    dpi: int = typer.Option(RENDER_DEFAULTS.dpi, "--dpi", help="Render DPI"),
):
    """
    Overlay the config bubble zones on top of a PDF page to verify placement.
    """
    try:
        overlay_config(
            config_path=config,
            input_path=input_pdf,
            out_image=out_image,
            dpi=dpi,
            pdf_renderer=pdf_renderer,
        )
    except Exception as e:
        rprint(f"[red]Visualization failed for {config}:[/red] {e}")
        raise typer.Exit(code=2)

    rprint(f"[green]Wrote:[/green] {out_image}")
    
    
# ---------------------- SCORE ----------------------
@app.command()
def grade(
    input_pdf: str = typer.Argument(..., help="Aligned scans PDF"),
    config: str = typer.Option(..., "--config", "-c", help="Config file (.yaml/.yml)"),
    key_txt: Optional[str] = typer.Option(None, "--key-txt", "-k",
        help="Answer key file (A/B/C/... one per line). If provided, only first len(key) questions are graded/output."),
    out_csv: str = typer.Option("results.csv", "--out-csv", "-o", help="Output CSV of per-student results"),
    out_annotated_dir: Optional[str] = typer.Option(None, "--out-annotated-dir", help="Directory to write annotated sheets"),
    annotate_all_cells: bool = typer.Option(False, "--annotate-all-cells", help="Draw every bubble in each row"),
    label_density: bool = typer.Option(False, "--label-density", help="Overlay % fill text at bubble centers"),
    dpi: int = typer.Option(RENDER_DEFAULTS.dpi, "--dpi", help="Scan/PDF render DPI"),
    min_fill: Optional[float] = typer.Option(None, "--min-fill", help=f"default {SCORING_DEFAULTS.min_fill}"),
    top2_ratio: Optional[float] = typer.Option(None, "--top2-ratio", help=f"default {SCORING_DEFAULTS.top2_ratio}"),
    min_score: Optional[float] = typer.Option(None, "--min-score", help=f"default {SCORING_DEFAULTS.min_score}"),
    min_abs: Optional[float] = typer.Option(None, "--min-abs", help=f"default {SCORING_DEFAULTS.min_abs}"),
    # (annotation flags you already added can stay)
):
    """
    Grade aligned scans using axis-based config.
    """
    # Build annotation overrides from CLI (if any) ... (your existing code)

    try:
        _ = load_config(config)
    except Exception as e:
        rprint(f"[red]Failed to load config {config}:[/red] {e}")
        raise typer.Exit(code=2)

    try:
        scoring = apply_scoring_overrides(  # ← use centralized helper
            min_fill=min_fill if min_fill is not None else SCORING_DEFAULTS.min_fill,
            top2_ratio=top2_ratio if top2_ratio is not None else SCORING_DEFAULTS.top2_ratio,
            min_score=min_score if min_score is not None else SCORING_DEFAULTS.min_score,
            min_abs=min_abs if min_abs is not None else SCORING_DEFAULTS.min_abs,
        )

        grade_pdf(
            input_path=input_pdf,
            config_path=config,
            out_csv=out_csv,
            key_txt=key_txt,
            out_annotated_dir=out_annotated_dir,
            dpi=dpi,
            min_fill=scoring.min_fill,
            top2_ratio=scoring.top2_ratio,
            min_score=scoring.min_score,
            min_abs=scoring.min_abs,
            annotate_all_cells=annotate_all_cells,
            label_density=label_density,
            # annot=annot_obj  ← if your grade_pdf supports it; otherwise thread it in there too
        )
    except Exception as e:
        rprint(f"[red]Grading failed:[/red] {e}")
        raise typer.Exit(code=2)

    rprint(f"[green]Wrote results:[/green] {out_csv}")


# ------------------------------ STATS -------------------------------
@app.command()
def stats(
    input_csv: str = typer.Argument(..., help="Results CSV (from 'grade')"),
    output_csv: str = typer.Option("results_with_stats.csv", "--output-csv", "-o", help="Augmented CSV with summary rows"),
    item_pattern: str = typer.Option(r"^Q\d+$", "--item-pattern", help="Regex for item columns (default: ^Q\\d+$)"),
    percent: bool = typer.Option(True, "--percent/--proportion", help="Report difficulty as percent (0-100) or proportion (0-1)"),
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


# --------------------------- COMPRESS-PDF ----------------------------
@app.command("compress-pdf")
def compress_pdf_cmd(
    input_pdf: str = typer.Argument(..., help="Input PDF to compress"),
    output_pdf: Optional[str] = typer.Option(None, "--output-pdf", "-o", help="Output path; if omitted, overwrite input"),
    quality: str = typer.Option("ebook", "--quality", help="Ghostscript -dPDFSETTINGS: screen|ebook|printer|prepress"),
    quiet: bool = typer.Option(True, "--quiet/--no-quiet", help="Suppress Ghostscript output"),
):
    """
    Compress a PDF using Ghostscript if installed (macOS: `brew install ghostscript`).
    """
    gs = "gs"
    args = [
        gs, "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        f"-dPDFSETTINGS=/{quality}",
        "-dNOPAUSE",
        "-dBATCH",
    ]
    if quiet:
        args.append("-dQUIET")

    in_path = Path(input_pdf).expanduser().resolve()
    if output_pdf:
        out_path = Path(output_pdf).expanduser().resolve()
    else:
        out_path = in_path  # overwrite

    if out_path == in_path:
        tmp_out = in_path.with_suffix(".compressed.pdf")
        args.extend(["-sOutputFile=" + str(tmp_out), str(in_path)])
        try:
            rprint(f"[cyan]Running:[/cyan] {' '.join(args)}")
            subprocess.run(args, check=True)
            tmp_out.replace(in_path)
            rprint(f"[green]Compressed (inplace):[/green] {in_path}")
        except FileNotFoundError:
            rprint("[red]Ghostscript ('gs') not found. Install it (e.g., `brew install ghostscript`).[/red]")
            raise typer.Exit(code=3)
        except subprocess.CalledProcessError as e:
            rprint(f"[red]Ghostscript failed:[/red] {e}")
            raise typer.Exit(code=4)
    else:
        args.extend(["-sOutputFile=" + str(out_path), str(in_path)])
        try:
            rprint(f"[cyan]Running:[/cyan] {' '.join(args)}")
            subprocess.run(args, check=True)
            rprint(f"[green]Compressed ->[/green] {out_path}")
        except FileNotFoundError:
            rprint("[red]Ghostscript ('gs') not found. Install it (e.g., `brew install ghostscript`).[/red]")
            raise typer.Exit(code=3)
        except subprocess.CalledProcessError as e:
            rprint(f"[red]Ghostscript failed:[/red] {e}")
            raise typer.Exit(code=4)


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

def _parse_bgr_csv(s: Optional[str]):
    if not s:
        return None
    try:
        parts = [int(x.strip()) for x in s.split(',')]
        if len(parts) != 3:
            raise ValueError
        return tuple(parts)
    except Exception:
        raise typer.BadParameter(f"Invalid BGR CSV: {s}")
