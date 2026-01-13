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
    )
except Exception:  # pragma: no cover
    SCORING_DEFAULTS = FEAT_DEFAULTS = MATCH_DEFAULTS = EST_DEFAULTS = ALIGN_DEFAULTS = RENDER_DEFAULTS = None

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
    """
    Initialize and display a working-directory selector.

    Default: the directory you launched `streamlit run` from (os.getcwd()).
    User can override via a text box in the sidebar.
    Also provides a click-to-browse folder picker (in-app), since Streamlit
    cannot open a native OS folder dialog by default.
    """
    global WORKDIR

    default_dir = Path(os.getcwd()).expanduser()

    # Seed session_state on first run
    if "workdir" not in st.session_state:
        st.session_state["workdir"] = str(default_dir)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Working directory")

    # 1) Typed input
    dir_str = st.sidebar.text_input(
        "Set directory for output/temp files",
        value=st.session_state["workdir"],
        help="Type/paste a path, or use Browse below to select a folder.",
    )
    if dir_str:
        st.session_state["workdir"] = dir_str

    # 2) In-app folder browser state
    if "workdir_browse_cursor" not in st.session_state:
        st.session_state["workdir_browse_cursor"] = str(default_dir)

    typed_path = Path(st.session_state["workdir"]).expanduser()

    # If typed workdir is valid, keep browse cursor aligned to it.
    if typed_path.exists() and typed_path.is_dir():
        st.session_state["workdir_browse_cursor"] = str(typed_path)

    with st.sidebar.expander("Browse for folder", expanded=False):
        cursor = Path(st.session_state["workdir_browse_cursor"]).expanduser()

        if not cursor.exists() or not cursor.is_dir():
            cursor = default_dir
            st.session_state["workdir_browse_cursor"] = str(cursor)

        st.caption(f"Browsing: `{cursor}`")

        cols = st.columns(2)
        if cols[0].button("Up one level", use_container_width=True, key="workdir_up"):
            st.session_state["workdir_browse_cursor"] = str(cursor.parent)
            st.rerun()

        if cols[1].button("Use this folder", use_container_width=True, key="workdir_use_this"):
            st.session_state["workdir"] = str(cursor)
            st.rerun()

        subdirs = _safe_list_subdirs(cursor)
        if not subdirs:
            st.info("No subfolders found here (or access is restricted).")
        else:
            choices = ["./ (this folder)"] + [p.name for p in subdirs]
            pick = st.selectbox("Select a subfolder", choices, key="workdir_pick")

            if st.button("Enter selected folder", use_container_width=True, key="workdir_enter"):
                new_cursor = cursor if pick == "./ (this folder)" else (cursor / pick)
                st.session_state["workdir_browse_cursor"] = str(new_cursor)
                st.rerun()

    # 3) Validate typed path and warn if needed (but keep your create-if-missing behavior)
    typed_path = Path(st.session_state["workdir"]).expanduser()
    if typed_path.exists() and not typed_path.is_dir():
        st.sidebar.warning(
            "The path you entered exists but is not a directory (it looks like a file). "
            "Please choose a folder path instead."
        )

    # If it does not exist, warn that it will be created.
    # If it exists and is a directory, no warning needed.
    if not typed_path.exists():
        st.sidebar.warning(
            "This folder does not exist yet. MarkShark will create it when you run a job (and it will be used for outputs)."
        )

    # Finalize WORKDIR and create it (current behavior)
    WORKDIR = typed_path
    WORKDIR.mkdir(parents=True, exist_ok=True)

    st.sidebar.caption(f"Currently using: {WORKDIR}")
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
page = st.sidebar.radio("Select below", ["1) Align scans", "2) Score", "3) Stats", "4) Bblmap visualizer", "5) Help"])

# Initialize / show working directory selector
_init_workdir()

# ===================== 1) ALIGN SCANS =====================
if page.startswith("1"):
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



# ===================== 5) HELP =====================
else:
    st.header("Help and CLI reference")
    st.markdown("""
**Typical workflow**
1. Align scans (raw student scans PDF + template PDF) to create an aligned PDF
2. Grade the aligned PDF (aligned PDF + bubblemap YAML, optional key.txt) to get results.csv
3. Compute stats from results.csv (item analysis, KR-20, plots)
4. (Optional) Visualize your bubblemap overlay to verify bubble ROI placement

If the GUI is missing something, the CLI is always the single source of truth.
""")

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
