
#!/usr/bin/env python3
"""
Streamlit GUI wrapper for the MarkSharkOMR CLI.
This app shells out to the Typer commands (align, visualize, grade, stats),
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

# --------------------- Working directory handling ---------------------
WORKDIR: Path | None = None

def _init_workdir() -> Path:
    """
    Initialize and display a working-directory selector.

    Default: the directory you launched `streamlit run` from (os.getcwd()).
    User can override via a text box in the sidebar.
    """
    global WORKDIR

    # default is the launch directory
    default_dir = Path(os.getcwd())

    # seed session_state if first run
    if "workdir" not in st.session_state:
        st.session_state["workdir"] = str(default_dir)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Working directory")

    dir_str = st.sidebar.text_input(
        "Set directory for output/temp files",
        value=st.session_state["workdir"],
    )

    # update state if user changed it
    if dir_str:
        st.session_state["workdir"] = dir_str

    WORKDIR = Path(st.session_state["workdir"]).expanduser()
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
st.sidebar.image("src/markshark/assets/shark.png")
st.sidebar.title("MarkShark 1.0")
page = st.sidebar.radio("Select below", ["1) Align scans", "2) Grade", "3) Stats", "4) Config visualizer"])

# Initialize / show working directory selector
_init_workdir()

# ===================== 1) ALIGN SCANS =====================
if page.startswith("1"):
    st.header("Align raw student scans to your template")

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
        out_pdf_name = st.text_input("Output aligned PDF name", value="aligned_scans.pdf")
        method = st.selectbox("Alignment method", ["auto", "aruco", "feature"], index=1)
        dpi = st.number_input("Render DPI", min_value=72, max_value=600, value=150, step=1)
        keep_intermediates = st.checkbox("Keep debug intermediates", value=True)

    with colB:
        st.markdown("ArUco mark alignment parameters")
        min_markers = st.number_input("Min ArUco markers to accept", min_value=0, max_value=32, value=0, step=1)
        dict_name = st.text_input("ArUco dictionary (if aruco)", value="DICT_4X4_50")

        st.markdown("---")
        st.markdown("Align parameters without ArUco markers")
        ransac = st.number_input("RANSAC reprojection threshold", min_value=0.1, max_value=20.0, value=2.0, step=0.1)
        confidence = st.number_input("USAC confidence (0–1)", min_value=0.10, max_value=1.00, value=0.99, step=0.01, format="%.2f")
        orb_nfeatures = st.number_input("ORB nfeatures", min_value=200, max_value=20000, value=3000, step=100)
        match_ratio = st.number_input("Match ratio (Lowe)", min_value=0.50, max_value=0.99, value=0.75, step=0.01, format="%.2f")

        st.markdown("---")
        first_page = st.number_input("First page to align in your raw scans (0 = auto)", min_value=0, value=0, step=1)
        last_page = st.number_input("Last page to align in your raw scans (0 = auto)", min_value=0, value=0, step=1)
        template_page = st.number_input("Template page (use if your template pdf has multiple pages)", min_value=1, value=1, step=1)

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
                "--method", method,
                "--template-page", str(int(template_page)),
                "--ransac", str(float(ransac)),
                "--confidence", str(float(confidence)),
                "--orb-nfeatures", str(int(orb_nfeatures)),
                "--match-ratio", str(float(match_ratio)),
                "--min-markers", str(int(min_markers)),
            ]
            if keep_intermediates:
                args += ["--keep-intermediates"]
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

                if keep_intermediates:
                    dbg = out_dir / "intermediates"
                    if dbg.exists():
                        st.download_button(
                            "Download intermediates.zip",
                            data=_zip_dir_to_bytes(dbg),
                            file_name="intermediates.zip",
                        )
            except Exception as e:
                status.error(f"Error during alignment: {e}")

# ===================== 2) GRADE =====================
elif page.startswith("2"):
    st.header("Grade aligned scans")
    # Top-of-page controls and status
    top_col1, top_col2 = st.columns([1, 3])
    with top_col1:
        run_score_clicked = st.button("Score/Grade")
    with top_col2:
        score_status = st.empty()  # all errors/updates will appear here

    st.divider()

    colA, colB = st.columns(2)
    with colA:
        aligned = _tempfile_from_uploader("Aligned scans PDF", "grade_pdf", types=("pdf",))
        config = _tempfile_from_uploader("Config (YAML)", "grade_cfg", types=("yaml","yml"))
        key_txt = _tempfile_from_uploader("Key TXT (optional)", "grade_key", types=("txt",))
        out_csv_name = st.text_input("Output results CSV", value="results.csv")
        out_ann_dir = st.text_input("Annotated pages directory (optional)", value="")
    with colB:
        annotate_all = st.checkbox("Annotate all cells", value=True)
        label_density = st.checkbox("Label density diagnostics", value=True)
        dpi = st.number_input("Render DPI", min_value=72, max_value=600, value=150, step=1)
        min_fill = st.text_input("min-fill (blank for default)", value="")
        top2_ratio = st.text_input("top2-ratio (blank for default)", value="")
        min_score = st.text_input("min-score (blank for default)", value="")
        min_abs = st.text_input("min-abs (blank for default)", value="")

    if run_score_clicked:
        if not aligned or not config:
            st.error("Please upload aligned PDF and config.")
        else:
            base = WORKDIR or Path(os.getcwd())
            out_dir = Path(tempfile.mkdtemp(prefix="grade_", dir=str(base)))
            out_csv = out_dir / out_csv_name
            args = [
                "grade",
                str(aligned),
                "--config", str(config),
                "--out-csv", str(out_csv),
                "--dpi", str(int(dpi)),
            ]
            if key_txt:
                args += ["--key-txt", str(key_txt)]
            if out_ann_dir.strip():
                ann_dir = out_dir / out_ann_dir.strip()
                args += ["--out-annotated-dir", str(ann_dir)]
            if annotate_all:
                args += ["--annotate-all-cells"]
            if label_density:
                args += ["--label-density"]
            # Optional scoring thresholds (only pass if user provided)
            if min_fill.strip():
                args += ["--min-fill", min_fill.strip()]
            if top2_ratio.strip():
                args += ["--top2-ratio", top2_ratio.strip()]
            if min_score.strip():
                args += ["--min-score", min_score.strip()]
            if min_abs.strip():
                args += ["--min-abs", min_abs.strip()]

            with st.spinner("Grading via CLI..."):
                try:
                    out = _run_cli(args)
                    st.code(out or "Done.", language="bash")
                    _download_file_button("Download results.csv", out_csv)
                    # Offer annotated pages if produced
                    if out_ann_dir.strip():
                        ann_dir = out_dir / out_ann_dir.strip()
                        if ann_dir.exists():
                            st.download_button("Download annotated_pages.zip", data=_zip_dir_to_bytes(ann_dir), file_name="annotated_pages.zip")
                except Exception as e:
                    score_status.error(str(e))

# ===================== 3) STATS =====================
elif page.startswith("3"):
    st.header("3) Item/exam statistics")
    colA, colB = st.columns(2)
    with colA:
        in_csv = _tempfile_from_uploader("Results CSV (from grade)", "stats_csv", types=("csv",))
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

# ===================== 4) VISUALIZE CONFIG =====================
else:
    st.header("4) Visualize config overlay")
    colA, colB = st.columns(2)
    with colA:
        pdf = _tempfile_from_uploader("Template or aligned PDF (single page preferred)", "viz_pdf", types=("pdf",))
        config = _tempfile_from_uploader("Config (YAML)", "viz_cfg", types=("yaml","yml"))
        out_image_name = st.text_input("Output image name", value="config_overlay.png")
    with colB:
        dpi = st.number_input("Render DPI", min_value=72, max_value=600, value=300, step=1)

    if st.button("Render overlay"):
        if not pdf or not config:
            st.error("Please upload a PDF and a config.")
        else:
            base = WORKDIR or Path(os.getcwd())
            out_dir = Path(tempfile.mkdtemp(prefix="viz_", dir=str(base)))
            out_img = out_dir / out_image_name
            args = [
                "visualize",
                str(pdf),
                "--config", str(config),
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


if __name__ == "__main__":
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "0")
    st.write("Run with:  streamlit run app_streamlit_cliwrap.py")
