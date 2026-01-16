#!/usr/bin/env python3
"""
Streamlit GUI wrapper for the MarkSharkOMR CLI.
This app shells out to the Typer commands (align, score, stats, visualize),
so the GUI stays in sync with the single source of truth: the CLI + defaults.py.
"""
from __future__ import annotations

import os
import io
import zipfile
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List

import streamlit as st

# Optional, pull defaults from MarkShark so GUI matches CLI defaults.
try:
    from markshark.defaults import (
        SCORING_DEFAULTS,
        FEAT_DEFAULTS,
        MATCH_DEFAULTS,
        EST_DEFAULTS,
        ALIGN_DEFAULTS,
        RENDER_DEFAULTS,
        TEMPLATE_DEFAULTS,
    )
    from markshark.template_manager import TemplateManager, BubbleSheetTemplate
except Exception:  # pragma: no cover
    SCORING_DEFAULTS = FEAT_DEFAULTS = MATCH_DEFAULTS = EST_DEFAULTS = ALIGN_DEFAULTS = RENDER_DEFAULTS = TEMPLATE_DEFAULTS = None
    TemplateManager = BubbleSheetTemplate = None

def _dflt(obj, attr: str, fallback):
    """Best-effort defaults helper when markshark.defaults is unavailable."""
    if obj is None:
        return fallback
    return getattr(obj, attr, fallback)

# --------------------- Working directory handling ---------------------
WORKDIR: Path | None = None

def _safe_list_subdirs(p: Path, max_entries: int = 300) -> list[Path]:
    try:
        subdirs = [x for x in p.iterdir() if x.is_dir()]
        subdirs.sort(key=lambda x: x.name.lower())
        return subdirs[:max_entries]
    except Exception:
        return []

def _init_workdir() -> Path:
    global WORKDIR
    default_dir = Path(os.getcwd()).expanduser()

    if "workdir" not in st.session_state:
        st.session_state["workdir"] = str(default_dir)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Working directory")

    # Text input with better help text
    dir_str = st.sidebar.text_input(
        "Working directory path",
        value=st.session_state["workdir"],
        help="💡 Tip: Copy/paste a path, or use the folder browser below",
    )
    if dir_str:
        st.session_state["workdir"] = dir_str

    # Simplified browser with common locations
    with st.sidebar.expander("📁 Browse for folder", expanded=False):
        # Quick access to common locations
        st.caption("**Quick locations:**")
        col1, col2 = st.columns(2)
        
        if col1.button("🏠 Home", use_container_width=True):
            st.session_state["workdir"] = str(Path.home())
            st.rerun()
        
        if col2.button("💼 Desktop", use_container_width=True):
            desktop = Path.home() / "Desktop"
            if desktop.exists():
                st.session_state["workdir"] = str(desktop)
                st.rerun()
        
        col3, col4 = st.columns(2)
        
        if col3.button("📄 Documents", use_container_width=True):
            docs = Path.home() / "Documents"
            if docs.exists():
                st.session_state["workdir"] = str(docs)
                st.rerun()
        
        if col4.button("⬇️ Downloads", use_container_width=True):
            downloads = Path.home() / "Downloads"
            if downloads.exists():
                st.session_state["workdir"] = str(downloads)
                st.rerun()
        
        st.divider()
        
        # Current location browser
        if "workdir_browse_cursor" not in st.session_state:
            st.session_state["workdir_browse_cursor"] = st.session_state["workdir"]
        
        cursor = Path(st.session_state["workdir_browse_cursor"]).expanduser()
        
        if not cursor.exists() or not cursor.is_dir():
            cursor = default_dir
            st.session_state["workdir_browse_cursor"] = str(cursor)
        
        st.caption(f"**Current location:**")
        st.code(str(cursor), language=None)
        
        # Navigation buttons
        cols = st.columns(2)
        if cols[0].button("⬆️ Up one level", use_container_width=True):
            st.session_state["workdir_browse_cursor"] = str(cursor.parent)
            st.rerun()
        
        if cols[1].button("✅ Use this folder", use_container_width=True, type="primary"):
            st.session_state["workdir"] = str(cursor)
            st.rerun()
        
        # Subdirectories list
        subdirs = _safe_list_subdirs(cursor) or []
        subdirs = [d for d in subdirs if not d.name.startswith(".")]
        if subdirs:
            st.caption("**Subfolders:**")
            for subdir in subdirs[:15]:  # Limit to 10 for cleaner UI
                
                if st.button(f"📁 {subdir.name}", use_container_width=True, key=f"nav_{subdir.name}"):
                    st.session_state["workdir_browse_cursor"] = str(subdir)
                    st.rerun()
            
            if len(subdirs) > 15:
                st.caption(f"... and {len(subdirs) - 15} more folders")
        else:
            st.info("No subfolders here")

    # Validation
    typed_path = Path(st.session_state["workdir"]).expanduser()
    if not typed_path.exists():
        st.sidebar.warning("⚠️ This folder doesn't exist yet. It will be created when needed.")
    elif not typed_path.is_dir():
        st.sidebar.error("❌ This path is not a folder!")

    WORKDIR = typed_path
    WORKDIR.mkdir(parents=True, exist_ok=True)
    
    st.sidebar.success(f"✅ Using: `{WORKDIR.name}`")
    return WORKDIR
    
    
st.set_page_config(page_title="MarkShark (GUI)", layout="wide")

# --------------------- Utilities ---------------------
def _tempfile_from_uploader(label: str, key: str, types=("pdf","yaml","yml","txt","csv","png","jpg","jpeg")) -> Optional[Path]:
    up = st.file_uploader(label, type=list(types), key=key)
    if not up:
        return None
    suffix = Path(up.name).suffix or ".bin"
    p = Path(tempfile.mkdtemp()) / f"upload_{key}{suffix}"
    p.write_bytes(up.getbuffer())
    st.caption(f"Saved: {p}")
    return p

def _run_cli(args: List[str]) -> str:
    """
    Run the MarkShark CLI. Prefer console script `markshark` if on PATH,
    else fallback to `python -m markshark.cli`.
    Returns combined stdout/stderr (raises on non-zero).
    """
    # First try console script
    cmds = [
        ["markshark"] + args,
        [sys.executable, "-m", "markshark.cli"] + args,
    ]
    last_err = None
    for cmd in cmds:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            out = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode != 0:
                last_err = RuntimeError(out.strip() or f"Non-zero exit: {proc.returncode}")
                continue
            return out
        except FileNotFoundError as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError("Unknown CLI invocation error")

def _download_file_button(label: str, path: Path):
    if path.exists():
        st.download_button(label, data=path.read_bytes(), file_name=path.name)

def _zip_dir_to_bytes(dir_path: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # sort for deterministic order; skip hidden files/dirs
        for p in sorted(dir_path.rglob("*")):
            rel = p.relative_to(dir_path)
            parts_hidden = any(part.startswith(".") for part in rel.parts)
            if parts_hidden or not p.is_file():
                continue
            zf.write(p, rel)
    buf.seek(0)
    return buf.read()

# --------------------- Sidebar ---------------------
# image_url = "https://github.com/navarrew/markshark/blob/main/images/shark.png" 
# st.sidebar.image(image_url, caption="MarkShark Logo", use_column_width=True)
st.sidebar.title("MarkShark Beta")
page = st.sidebar.radio("Select below", [
    "0) Quick grade (Align + Score)",
    "1) Align scans",
    "2) Score",
    "3) Stats",
    "4) Bblmap visualizer",
    "5) Template manager",
    "6) Help"
])

# Initialize / show working directory selector
_init_workdir()

# ===================== 0) QUICK GRADE (UNIFIED WORKFLOW) =====================
if page.startswith("0"):
    st.header("Quick Grade: Align + Score in One Step")
    st.markdown("""
    Upload your **scanned answer sheets** and **answer key**, select a template, and MarkShark will:
    1. Align the scans to the template
    2. Score them automatically
    """)
    
    # Top-of-page controls
    top_col1, top_col2 = st.columns([1, 3])
    with top_col1:
        run_quick_grade = st.button("Run Quick Grade", type="primary")
    with top_col2:
        quick_status = st.empty()
    
    st.divider()
    
    # Initialize template manager
    template_manager = None
    if TemplateManager is not None:
        try:
            templates_dir = _dflt(TEMPLATE_DEFAULTS, "templates_dir", None)
            template_manager = TemplateManager(templates_dir)
        except Exception as e:
            st.warning(f"Could not initialize template manager: {e}")
    
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Inputs")
        scans = _tempfile_from_uploader("Scanned answer sheets (PDF)", "quick_scans", types=("pdf",))
        key_txt = _tempfile_from_uploader("Answer key (TXT)", "quick_key", types=("txt",))
        
        # Template selection
        st.markdown("---")
        st.subheader("Template Selection")
        
        template_choice = None
        custom_template_pdf = None
        custom_bublmap = None
        
        if template_manager is not None:
            templates = template_manager.scan_templates()
            
            if templates:
                template_names = ["(Upload custom files)"] + [str(t) for t in templates]
                selected_name = st.selectbox("Select bubble sheet template", template_names)
                
                if selected_name != "(Upload custom files)":
                    # Find the selected template
                    for t in templates:
                        if str(t) == selected_name:
                            template_choice = t
                            break
                    
                    if template_choice:
                        st.success(f"Using template: **{template_choice.display_name}**")
                        if template_choice.num_questions:
                            st.caption(f"Questions: {template_choice.num_questions} | Choices: {template_choice.num_choices or 'N/A'}")
            else:
                st.info("No templates found. Upload custom files below or add templates to the templates directory.")
        
        # Custom upload option
        if template_choice is None:
            st.markdown("**Upload custom template files:**")
            custom_template_pdf = _tempfile_from_uploader("Master template PDF", "quick_template_pdf", types=("pdf",))
            custom_bublmap = _tempfile_from_uploader("Bubblemap YAML", "quick_bubblemap", types=("yaml", "yml"))
    
    with colB:
        st.subheader("Options")
        
        # Output names
        out_csv_name = st.text_input("Results CSV name", value="quick_grade_results.csv")
        out_pdf_name = st.text_input("Annotated PDF name", value="quick_grade_annotated.pdf")
        
        st.markdown("---")
        dpi = st.number_input("Render DPI", min_value=72, max_value=600, value=int(_dflt(RENDER_DEFAULTS, "dpi", 300)), step=1)
        
        # Scoring options
        with st.expander("Scoring options", expanded=False):
            annotate_all = st.checkbox("Annotate all bubbles", value=True)
            label_density = st.checkbox("Show % fill labels", value=True)
            auto_thresh = st.checkbox("Auto-calibrate threshold", value=_dflt(SCORING_DEFAULTS, "auto_calibrate_thresh", True))
            verbose_thresh = st.checkbox("Verbose threshold calibration", value=True)
            
            min_fill = st.text_input("Min fill", value="", placeholder=str(_dflt(SCORING_DEFAULTS, "min_fill", "")))
            min_score = st.text_input("Min score", value="", placeholder=str(_dflt(SCORING_DEFAULTS, "min_score", "")))
            top2_ratio = st.text_input("Top2 ratio", value="", placeholder=str(_dflt(SCORING_DEFAULTS, "top2_ratio", "")))
        
        # Alignment options
        with st.expander("Alignment options", expanded=False):
            align_method = st.selectbox("Alignment method", ["auto", "aruco", "feature"], index=0)
            min_markers = st.number_input("Min ArUco markers", min_value=0, max_value=32, value=int(_dflt(ALIGN_DEFAULTS, "min_aruco", 4)), step=1)
    
    # Run quick grade workflow
    if run_quick_grade:
        # Validate inputs
        if not scans:
            quick_status.error("Please upload scanned answer sheets PDF")
        elif template_choice is None and (not custom_template_pdf or not custom_bublmap):
            quick_status.error("Please select a template or upload custom template files")
        else:
            # Determine template files to use
            if template_choice:
                template_pdf = template_choice.template_pdf_path
                bublmap = template_choice.bubblemap_yaml_path
            else:
                template_pdf = custom_template_pdf
                bublmap = custom_bublmap
            
            base = WORKDIR or Path(os.getcwd())
            work_dir = Path(tempfile.mkdtemp(prefix="quick_grade_", dir=str(base)))
            
            try:
                # Step 1: Align
                quick_status.info("Step 1/2: Aligning scans to template...")
                aligned_pdf = work_dir / "aligned_scans.pdf"
                
                align_args = [
                    "align",
                    str(scans),
                    "--template", str(template_pdf),
                    "--out-pdf", str(aligned_pdf),
                    "--dpi", str(int(dpi)),
                    "--align-method", align_method,
                    "--min-markers", str(min_markers),
                ]
                
                with st.spinner("Aligning scans..."):
                    align_out = _run_cli(align_args)
                    quick_status.success("✓ Alignment complete")
                
                # Step 2: Score
                quick_status.info("Step 2/2: Scoring aligned sheets...")
                out_csv = work_dir / out_csv_name
                out_pdf = work_dir / out_pdf_name
                
                score_args = [
                    "score",
                    str(aligned_pdf),
                    "--bublmap", str(bublmap),
                    "--out-csv", str(out_csv),
                    "--out-pdf", out_pdf_name,
                    "--dpi", str(int(dpi)),
                ]
                
                if key_txt:
                    score_args += ["--key-txt", str(key_txt)]
                if annotate_all:
                    score_args += ["--annotate-all-cells"]
                if label_density:
                    score_args += ["--label-density"]
                if not auto_thresh:
                    score_args += ["--no-auto-thresh"]
                if verbose_thresh:
                    score_args += ["--verbose-thresh"]
                if min_fill.strip():
                    score_args += ["--min-fill", min_fill.strip()]
                if top2_ratio.strip():
                    score_args += ["--top2-ratio", top2_ratio.strip()]
                if min_score.strip():
                    score_args += ["--min-score", min_score.strip()]
                
                with st.spinner("Scoring sheets..."):
                    score_out = _run_cli(score_args)
                    quick_status.success("✅ Quick Grade complete!")
                
                # Display results
                st.success("Processing complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    _download_file_button("📄 Download Results CSV", out_csv)
                with col2:
                    _download_file_button("📑 Download Annotated PDF", out_pdf)
                with col3:
                    _download_file_button("📋 Download Aligned Scans", aligned_pdf)
                
                # Show output logs
                with st.expander("View processing logs", expanded=False):
                    st.text("Alignment output:")
                    st.code(align_out or "Done.", language="bash")
                    st.text("Scoring output:")
                    st.code(score_out or "Done.", language="bash")
                    
            except Exception as e:
                quick_status.error(f"Error: {str(e)}")
                st.exception(e)

# ===================== 1) ALIGN SCANS =====================
elif page.startswith("1"):
    st.header("Align raw scans to your template bubblesheet")

    # Top-of-page controls and status
    top_col1, top_col2 = st.columns([1, 3])
    with top_col1:
        run_align_clicked = st.button("Run Alignment")
    with top_col2:
        status = st.empty()  # all errors/updates will appear here

    st.divider()

    colA, colB = st.columns(2)
    with colA:
        scans = _tempfile_from_uploader("Raw student scans (PDF)", "align_scans", types=("pdf",))
        template = _tempfile_from_uploader("Template bubble sheet (PDF)", "align_template", types=("pdf",))
        method = st.selectbox("Alignment method", ["auto", "aruco", "feature"], index=0)
        dpi = st.number_input("Render DPI", min_value=72, max_value=600, value=int(_dflt(RENDER_DEFAULTS, "dpi", 150)), step=1)

    with colB:
        out_pdf_name = st.text_input("Output aligned PDF name", value="aligned_scans.pdf")
        st.markdown("---")
        st.markdown("ArUco mark alignment parameters")
        min_markers = st.number_input("Min ArUco markers to accept", min_value=0, max_value=32, value=int(_dflt(ALIGN_DEFAULTS, "min_aruco", 0)), step=1)
        dict_name = st.text_input("ArUco dictionary (if aruco)", value=str(_dflt(ALIGN_DEFAULTS, "dict_name", "DICT_4X4_50")))

        st.markdown("---")
        st.markdown("Non-ArUco align parameters")
        ransac = st.number_input("RANSAC reprojection threshold", min_value=0.1, max_value=20.0, value=float(_dflt(EST_DEFAULTS, "ransac_thresh", 2.0)), step=0.1)
        orb_nfeatures = st.number_input("ORB nfeatures", min_value=200, max_value=20000, value=int(_dflt(FEAT_DEFAULTS, "orb_nfeatures", 3000)), step=100)
        match_ratio = st.number_input("Match ratio (Lowe)", min_value=0.50, max_value=0.99, value=float(_dflt(MATCH_DEFAULTS, "ratio_test", 0.75)), step=0.01, format="%.2f")

        with st.expander("Advanced (estimator and ECC)", expanded=False):
            estimator_method = st.selectbox(
                "Homography estimator method",
                ["auto", "ransac", "usac"],
                index=0,
                help="Maps to --estimator-method in the CLI (auto|ransac|usac).",
            )
            use_ecc = st.checkbox(
                "Enable ECC refinement",
                value=bool(_dflt(EST_DEFAULTS, "use_ecc", True)),
                help="Maps to --use-ecc/--no-use-ecc in the CLI.",
            )
            ecc_max_iters = st.number_input(
                "ECC max iterations",
                min_value=1,
                max_value=5000,
                value=int(_dflt(EST_DEFAULTS, "ecc_max_iters", 250)),
                step=10,
            )
            ecc_eps = st.number_input(
                "ECC epsilon",
                min_value=1e-7,
                max_value=1e-2,
                value=float(_dflt(EST_DEFAULTS, "ecc_eps", 1e-5)),
                step=1e-6,
                format="%.7f",
            )

        st.markdown("---")
        template_page = st.number_input("Template page (use if your template pdf has multiple pages)", min_value=1, value=1, step=1)
        first_page = st.number_input("First page to align in your raw scans (0 = auto)", min_value=0, value=0, step=1)
        last_page = st.number_input("Last page to align in your raw scans (0 = auto)", min_value=0, value=0, step=1)

    if run_align_clicked:
        if not scans or not template:
            status.error("Please upload scans and template.")
        else:
            base = WORKDIR or Path(os.getcwd())
            out_dir = Path(tempfile.mkdtemp(prefix="align_", dir=str(base)))
            out_pdf = out_dir / out_pdf_name
            args = [
                "align",
                str(scans),
                "--template", str(template),
                "--out-pdf", str(out_pdf),
                "--dpi", str(int(dpi)),
                "--align_method", method,
                "--estimator-method", estimator_method,
                "--template-page", str(int(template_page)),
                "--ransac", str(float(ransac)),
                "--orb-nfeatures", str(int(orb_nfeatures)),
                "--match-ratio", str(float(match_ratio)),
                "--min-markers", str(int(min_markers)),
            ]
            if dict_name.strip():
                args += ["--dict-name", dict_name.strip()]
            if first_page > 0:
                args += ["--first-page", str(int(first_page))]
            if last_page > 0:
                args += ["--last-page", str(int(last_page))]

            try:
                with st.spinner("Aligning via CLI..."):
                    out = _run_cli(args)
                status.success("Alignment finished.")
                st.code(out or "Done.", language="bash")

                _download_file_button("Download aligned_scans.pdf", out_pdf)

            except Exception as e:
                status.error(f"Error during alignment: {e}")

# ===================== 2) SCORE =====================
elif page.startswith("2"):
    st.header("Score aligned scans")
    # Top-of-page controls and status
    top_col1, top_col2 = st.columns([1, 3])
    with top_col1:
        run_score_clicked = st.button("Score")
    with top_col2:
        score_status = st.empty()  # all errors/updates will appear here

    st.divider()

    colA, colB = st.columns(2)
    with colA:
        aligned = _tempfile_from_uploader("Aligned scans PDF", "score_pdf", types=("pdf",))
        bublmap = _tempfile_from_uploader("Bubblemap (YAML)", "score_cfg", types=("yaml","yml"))
        key_txt = _tempfile_from_uploader("Key TXT (optional)", "score_key", types=("txt",))
    with colB:
        out_csv_name = st.text_input("Output results CSV", value="results.csv")
        scored_pdf_name = st.text_input("Annotated scored PDF filename", value="", placeholder=str(_dflt(SCORING_DEFAULTS, "out_pdf", "scored_scans.pdf")))
        out_ann_dir = st.text_input("Annotated png directory (optional)", value="", placeholder="Enter a folder name here for png files")
        annotate_all = st.checkbox("Annotate all cells", value=True)
        label_density = st.checkbox("Label density diagnostics", value=True)
        dpi = st.number_input("Render DPI", min_value=72, max_value=600, value=int(_dflt(RENDER_DEFAULTS, "dpi", 150)), step=1)
        st.markdown("---")
        st.markdown("Adjustments if scoring isn't working well")
        auto_thresh = st.checkbox(
            "Auto tune gray sensitivity per page",
            value=bool(_dflt(SCORING_DEFAULTS, "auto_calibrate_thresh", True)),
        )
        verbose_thresh = st.checkbox(
            "Show threshold calibration logs",
            value=bool(_dflt(SCORING_DEFAULTS, "verbose_calibration", False)),
        )
        fixed_thresh = st.text_input("Gray sensitivity (1-255: higher number = more sensitive to lighter shades)", value="", placeholder=str(_dflt(SCORING_DEFAULTS, "fixed_thresh", "")))
        min_fill = st.text_input("Minimum bubble area filled", value="", placeholder=str(_dflt(SCORING_DEFAULTS, "min_fill", "")))
        min_score = st.text_input("Minimum fill area difference between top two filled bubbles", value="", placeholder=str(_dflt(SCORING_DEFAULTS, "min_score", "")))
        top2_ratio = st.text_input("Minimum area fill ratio between 1st and 2nd most filled bubble", value="", placeholder=str(_dflt(SCORING_DEFAULTS, "top2_ratio", "")))
    if run_score_clicked:
        if not aligned or not bublmap:
            st.error("Please upload aligned PDF and bubblemap.")
        else:
            base = WORKDIR or Path(os.getcwd())
            out_dir = Path(tempfile.mkdtemp(prefix="score_", dir=str(base)))
            out_csv = out_dir / out_csv_name
            args = [
                "score",
                str(aligned),
                "--bublmap", str(bublmap),
                "--out-csv", str(out_csv),
                "--dpi", str(int(dpi)),
            ]
            if key_txt:
                args += ["--key-txt", str(key_txt)]
            if out_ann_dir.strip():
                ann_dir = out_dir / out_ann_dir.strip()
                args += ["--out-annotated-dir", str(ann_dir)]
            # Optional annotated PDF output name (only pass if user provided)
            if scored_pdf_name.strip():
                args += ["--out-pdf", scored_pdf_name.strip()]
            if annotate_all:
                args += ["--annotate-all-cells"]
            if label_density:
                args += ["--label-density"]
            if not auto_thresh:
                args += ["--no-auto-thresh"]
            if verbose_thresh:
                args += ["--verbose-thresh"]
            # Optional scoring thresholds (only pass if user provided)
            if fixed_thresh.strip():
                args += ["--fixed-thresh", fixed_thresh.strip()]
            if min_fill.strip():
                args += ["--min-fill", min_fill.strip()]
            if top2_ratio.strip():
                args += ["--top2-ratio", top2_ratio.strip()]
            if min_score.strip():
                args += ["--min-score", min_score.strip()]

            with st.spinner("Scoring tests via CLI..."):
                try:
                    out = _run_cli(args)
                    st.code(out or "Done.", language="bash")
                    _download_file_button("Download results.csv", out_csv)
                    # Offer annotated PDF if produced
                    pdf_name = scored_pdf_name.strip() or str(_dflt(SCORING_DEFAULTS, "out_pdf", "scored_scans.pdf"))
                    pdf_path = Path(pdf_name) if Path(pdf_name).is_absolute() else (out_dir / pdf_name)
                    _download_file_button(f"Download {pdf_path.name}", pdf_path)

                    # Offer annotated pages if produced
                    if out_ann_dir.strip():
                        ann_dir = out_dir / out_ann_dir.strip()
                        if ann_dir.exists():
                            st.download_button("Download annotated_pages.zip", data=_zip_dir_to_bytes(ann_dir), file_name="annotated_pages.zip")
                except Exception as e:
                    score_status.error(str(e))

# ===================== 3) STATS =====================
elif page.startswith("3"):
    st.header("Item/exam statistics")
    colA, colB = st.columns(2)
    with colA:
        in_csv = _tempfile_from_uploader("Results CSV (from score)", "stats_csv", types=("csv",))
        out_csv_name = st.text_input("Augmented CSV name", value="results_with_stats.csv")
        item_report_name = st.text_input("Item report CSV", value="item_analysis.csv")
        exam_stats_name = st.text_input("Exam stats CSV", value="exam_stats.csv")
        want_plots = st.checkbox("Generate item plots", value=False)
    with colB:
        percent = st.checkbox("Report difficulty as percent", value=True)
        label_col = st.text_input("Student label column", value="name")
        key_row_index = st.text_input("Key row index (blank=auto)", value="")
        answers_mode = st.selectbox("Answers mode", ["letters", "index"], index=0)
        key_label = st.text_input("Key label for auto-detect", value="KEY")
        decimals = st.number_input("Decimals", min_value=0, max_value=6, value=3, step=1)

    if st.button("Compute stats"):
        if not in_csv:
            st.error("Please upload the results CSV.")
        else:
            base = WORKDIR or Path(os.getcwd())
            out_dir = Path(tempfile.mkdtemp(prefix="stats_", dir=str(base)))
            out_csv = out_dir / out_csv_name
            item_csv = out_dir / item_report_name
            exam_csv = out_dir / exam_stats_name
            plots_dir = out_dir / "item_plots" if want_plots else None
            args = [
                "stats",
                str(in_csv),
                "--output-csv", str(out_csv),
                "--item-report-csv", str(item_csv),
                "--exam-stats-csv", str(exam_csv),
                "--label-col", label_col,
                "--answers-mode", answers_mode,
                "--key-label", key_label,
                "--decimals", str(int(decimals)),
            ]
            if percent:
                args += ["--percent"]
            else:
                args += ["--proportion"]
            if plots_dir:
                args += ["--plots-dir", str(plots_dir)]
            if key_row_index.strip():
                args += ["--key-row-index", key_row_index.strip()]

            with st.spinner("Computing stats via CLI..."):
                try:
                    out = _run_cli(args)
                    st.code(out or "Done.", language="bash")
                    _download_file_button("Download results_with_stats.csv", out_csv)
                    _download_file_button("Download item_analysis.csv", item_csv)
                    _download_file_button("Download exam_stats.csv", exam_csv)
                    if plots_dir and Path(plots_dir).exists():
                        st.download_button("Download item_plots.zip", data=_zip_dir_to_bytes(Path(plots_dir)), file_name="item_plots.zip")
                except Exception as e:
                    st.error(str(e))

# ===================== 4) VISUALIZE BUBBLEMAP ====================
elif page.startswith("4"):
    st.header("View your bubblemap.yaml on your template")
    colA, colB = st.columns(2)
    with colA:
        pdf = _tempfile_from_uploader("Template or aligned PDF (single page preferred)", "viz_pdf", types=("pdf",))
        bublmap = _tempfile_from_uploader("Bubblemap (YAML)", "viz_cfg", types=("yaml","yml"))
    with colB:
        out_image_name = st.text_input("Output image name", value="bubblemap_overlay.png")
        dpi = st.number_input("Render DPI", min_value=72, max_value=600, value=300, step=1)

    if st.button("Render overlay"):
        if not pdf or not bublmap:
            st.error("Please upload a PDF and a bubblemap.")
        else:
            base = WORKDIR or Path(os.getcwd())
            out_dir = Path(tempfile.mkdtemp(prefix="viz_", dir=str(base)))
            out_img = out_dir / out_image_name
            args = [
                "visualize",
                str(pdf),
                "--bublmap", str(bublmap),
                "--out-image", str(out_img),
                "--dpi", str(int(dpi)),
            ]
            with st.spinner("Rendering via CLI..."):
                try:
                    out = _run_cli(args)
                    st.code(out or "Done.", language="bash")
                    _download_file_button("Download overlay image", out_img)
                except Exception as e:
                    st.error(str(e))



# ===================== 5) MANAGE TEMPLATES =====================
elif page.startswith("5"):
    st.header("Manage Bubble Sheet Templates")
    
    # Initialize template manager
    template_manager = None
    if TemplateManager is not None:
        try:
            templates_dir_path = _dflt(TEMPLATE_DEFAULTS, "templates_dir", None)
            template_manager = TemplateManager(templates_dir_path)
            
            st.info(f"**Templates directory:** `{template_manager.templates_dir}`")
            st.caption("You can change this by setting the MARKSHARK_TEMPLATES_DIR environment variable or editing defaults.py")
        except Exception as e:
            st.error(f"Could not initialize template manager: {e}")
    else:
        st.error("Template manager not available. Please ensure markshark.template_manager is installed.")
    
    if template_manager:
        # Refresh button
        if st.button("🔄 Refresh Template List"):
            template_manager._templates_cache = None
            st.rerun()
        
        st.divider()
        
        # Display existing templates
        templates = template_manager.scan_templates(force_refresh=False)
        
        if templates:
            st.subheader(f"Available Templates ({len(templates)})")
            
            for template in templates:
                with st.expander(f"📋 {template.display_name}", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Template ID:** `{template.template_id}`")
                        if template.description:
                            st.markdown(f"**Description:** {template.description}")
                        if template.num_questions:
                            st.markdown(f"**Questions:** {template.num_questions}")
                        if template.num_choices:
                            st.markdown(f"**Choices per question:** {template.num_choices}")
                        
                        st.markdown(f"**PDF:** `{template.template_pdf_path.name}`")
                        st.markdown(f"**YAML:** `{template.bubblemap_yaml_path.name}`")
                    
                    with col2:
                        # Validate template
                        is_valid, errors = template_manager.validate_template(template)
                        if is_valid:
                            st.success("✅ Valid")
                        else:
                            st.error("❌ Invalid")
                            for error in errors:
                                st.caption(f"• {error}")
                    
                    # Show full paths
                    with st.expander("Full paths"):
                        st.code(str(template.template_pdf_path))
                        st.code(str(template.bubblemap_yaml_path))
        else:
            st.warning("No templates found in the templates directory.")
        
        st.divider()
        
        # Add new template section
        st.subheader("Add New Template")
        
        with st.expander("Upload new template files", expanded=False):
            st.markdown("""
            To add a new template:
            1. Create a folder in the templates directory with a unique name (e.g., `my_custom_template`)
            2. Place two files in that folder:
               - `master_template.pdf` - The blank bubble sheet PDF
               - `bubblemap.yaml` - The bubble zone configuration
            3. Click "Refresh Template List" above
            
            Alternatively, use the form below to upload files and MarkShark will create the folder for you.
            """)
            
            new_template_id = st.text_input(
                "Template ID (folder name)",
                placeholder="e.g., my_50q_test",
                help="Use lowercase letters, numbers, and underscores only"
            )
            new_display_name = st.text_input(
                "Display Name",
                placeholder="e.g., My 50 Question Test",
                help="Human-readable name shown in dropdowns"
            )
            new_description = st.text_input(
                "Description (optional)",
                placeholder="e.g., 50 questions, 4 choices (A-D)"
            )
            
            new_pdf = st.file_uploader("Upload master template PDF", type=["pdf"], key="new_template_pdf")
            new_yaml = st.file_uploader("Upload bubblemap YAML", type=["yaml", "yml"], key="new_template_yaml")
            
            if st.button("Create Template"):
                if not new_template_id or not new_template_id.replace('_', '').isalnum():
                    st.error("Please provide a valid template ID (letters, numbers, underscores only)")
                elif not new_pdf or not new_yaml:
                    st.error("Please upload both PDF and YAML files")
                else:
                    try:
                        # Create template directory
                        template_dir = template_manager.templates_dir / new_template_id
                        if template_dir.exists():
                            st.error(f"Template '{new_template_id}' already exists!")
                        else:
                            template_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Save PDF
                            pdf_path = template_dir / "master_template.pdf"
                            pdf_path.write_bytes(new_pdf.getbuffer())
                            
                            # Load and update YAML with metadata
                            yaml_data = yaml.safe_load(new_yaml.getvalue())
                            if 'metadata' not in yaml_data:
                                yaml_data['metadata'] = {}
                            if new_display_name:
                                yaml_data['metadata']['display_name'] = new_display_name
                            if new_description:
                                yaml_data['metadata']['description'] = new_description
                            
                            # Save YAML
                            yaml_path = template_dir / "bubblemap.yaml"
                            with open(yaml_path, 'w') as f:
                                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
                            
                            st.success(f"✅ Template '{new_template_id}' created successfully!")
                            st.info("Click 'Refresh Template List' above to see your new template.")
                            
                    except Exception as e:
                        st.error(f"Error creating template: {e}")
        
        # Directory structure helper
        with st.expander("📁 Expected directory structure"):
            st.code("""
templates/
├── my_template_1/
│   ├── master_template.pdf
│   └── bubblemap.yaml
├── my_template_2/
│   ├── master_template.pdf
│   └── bubblemap.yaml
└── ...
            """, language="text")
            
            st.markdown("**bubblemap.yaml** can optionally include metadata:")
            st.code("""
metadata:
  display_name: "My 50 Question Test"
  description: "50 questions, 5 choices (A-E)"
  num_questions: 50
  num_choices: 5

answer_rows:
  # ... your bubble coordinates ...
            """, language="yaml")


# ===================== 6) HELP =====================
elif page.startswith("6"):
    st.header("Help and CLI reference")
    st.markdown("""
**New Unified Workflow (Recommended)**
- **Quick Grade**: Upload scanned answer sheets + answer key, select a template → get results instantly!

**Traditional Workflow (Advanced)**
1. **Manage Templates**: Set up your bubble sheet templates (once per template type)
2. **Align scans**: Align raw student scans to template PDF → create aligned PDF
3. **Score**: Grade the aligned PDF with bubblemap YAML and answer key → get results.csv
4. **Stats**: Compute item analysis, KR-20, plots from results.csv
5. **Visualize**: Verify bubblemap overlay on your template (for template creation)

**About Templates**
Templates combine the master bubble sheet PDF and its bubblemap YAML configuration.
Store them in: `{template_dir}`
Each template needs its own folder with:
- `master_template.pdf` 
- `bubblemap.yaml`

If the GUI is missing something, the CLI is always the single source of truth.
""".format(template_dir=template_manager.templates_dir if (template_manager and hasattr(template_manager, 'templates_dir')) else "templates/"))

    st.subheader("Show CLI help")
    topic = st.selectbox("Help topic", ["markshark", "align", "score", "stats", "visualize"], index=0)
    help_args = {
        "markshark": ["--help"],
        "align": ["align", "--help"],
        "score": ["score", "--help"],
        "stats": ["stats", "--help"],
        "visualize": ["visualize", "--help"],
    }

    @st.cache_data(show_spinner=False)
    def _cached_help(args: List[str]) -> str:
        return _run_cli(args)

    try:
        st.code(_cached_help(help_args[topic]) or "(no output)", language="text")
    except Exception as e:
        st.error(f"Could not run CLI help: {e}")

if __name__ == "__main__":
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "0")
    st.write("Run with:  streamlit run app_streamlit.py")
