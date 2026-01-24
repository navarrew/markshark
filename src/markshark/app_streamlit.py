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
import streamlit_antd_components as sac
import yaml  # For template creation
import platform

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
    from markshark.project_utils import (
        sanitize_project_name,
        create_project_structure,
        create_run_directory,
        find_projects,
        get_project_info,
        get_next_run_number,
    )
    from markshark.preferences import (
        get_preference,
        set_preference,
    )
except Exception:  # pragma: no cover
    SCORING_DEFAULTS = FEAT_DEFAULTS = MATCH_DEFAULTS = EST_DEFAULTS = ALIGN_DEFAULTS = RENDER_DEFAULTS = TEMPLATE_DEFAULTS = None
    TemplateManager = BubbleSheetTemplate = None
    sanitize_project_name = create_project_structure = create_run_directory = find_projects = get_project_info = get_next_run_number = None
    get_preference = set_preference = None

def _dflt(obj, attr: str, fallback):
    """Best-effort defaults helper when markshark.defaults is unavailable."""
    if obj is None:
        return fallback
    return getattr(obj, attr, fallback)

# --------------------- Working directory handling ---------------------
WORKDIR: Path | None = None


def _get_known_folder(name: str) -> Path | None:
    """Get platform-specific known folder path (cross-platform compatible)."""
    home = Path.home()
    candidate = home / name

    # On Windows, also check for OneDrive-redirected folders
    if platform.system() == "Windows":
        onedrive_candidate = home / "OneDrive" / name
        if onedrive_candidate.exists():
            return onedrive_candidate

    return candidate if candidate.exists() else None


def _build_folder_tree(base_path: Path, max_depth: int = 2, current_depth: int = 0) -> list:
    """
    Recursively build tree items from directory structure for sac.tree().

    Returns list of sac.TreeItem objects representing the folder hierarchy.
    Skips macOS protected directories to avoid privacy permission prompts.
    """
    if current_depth >= max_depth:
        return []

    # macOS protected directories that trigger permission prompts
    MACOS_PROTECTED = {
        "Music", "Movies", "Pictures", "Photos", "Library",
        "Mail", "Messages", "Contacts", "Calendars",
        "Application Support", "Caches",
    }

    items = []
    try:
        subdirs = sorted(
            [
                d for d in base_path.iterdir()
                if d.is_dir()
                and not d.name.startswith(".")
                and d.name not in MACOS_PROTECTED
            ],
            key=lambda x: x.name.lower()
        )[:25]  # Limit for performance

        for subdir in subdirs:
            # Only recurse if we haven't hit max depth
            children = []
            if current_depth < max_depth - 1:
                children = _build_folder_tree(subdir, max_depth, current_depth + 1)

            items.append(sac.TreeItem(
                label=subdir.name,
                icon="folder",
                children=children if children else None,
            ))
    except (PermissionError, OSError):
        pass

    return items


def _init_workdir() -> Path:
    """
    Initialize and render the project + working directory picker in the sidebar.

    Uses streamlit-antd-components for a modern, clean UI with:
    - Project selection first (most common action)
    - Persistent preferences for default working directory
    - Segmented quick-location buttons
    - Interactive folder tree (no page reloads on expand/collapse)
    """
    global WORKDIR

    # Load saved preferences
    saved_workdir = get_preference("default_workdir") if get_preference else None
    saved_project = get_preference("last_project") if get_preference else None

    # Determine default directory (preference > cwd)
    if saved_workdir and Path(saved_workdir).exists():
        default_dir = Path(saved_workdir)
    else:
        default_dir = Path(os.getcwd()).expanduser()

    # Initialize session state
    if "workdir" not in st.session_state:
        st.session_state["workdir"] = str(default_dir)
    if "tree_base" not in st.session_state:
        st.session_state["tree_base"] = str(default_dir)
    if "project_name" not in st.session_state:
        st.session_state["project_name"] = saved_project or ""

    # We need WORKDIR set before project section can reference it
    typed_path = Path(st.session_state["workdir"]).expanduser()
    WORKDIR = typed_path
    WORKDIR.mkdir(parents=True, exist_ok=True)

    st.sidebar.markdown("---")

    # ===================== PROJECT SECTION (First!) =====================
    st.sidebar.markdown("##### Project")

    with st.sidebar:
        # Project name input
        project_input = st.text_input(
            "Project name",
            value=st.session_state["project_name"],
            placeholder="e.g., BIO101 Final Exam",
            label_visibility="collapsed",
            key="project_input",
        )

        if project_input != st.session_state["project_name"]:
            st.session_state["project_name"] = project_input
            # Save to preferences
            if set_preference:
                set_preference("last_project", project_input)

        # Show project status
        if st.session_state["project_name"].strip():
            sanitized = sanitize_project_name(st.session_state["project_name"]) if sanitize_project_name else st.session_state["project_name"]
            project_dir = WORKDIR / sanitized

            if project_dir.exists() and get_project_info:
                info = get_project_info(project_dir)
                run_text = f"{info['num_runs']} run{'s' if info['num_runs'] != 1 else ''}" if info['num_runs'] > 0 else "No runs yet"
                sac.alert(
                    label=sanitized,
                    description=run_text,
                    icon="folder2",
                    color="info",
                    size="sm",
                )
            else:
                sac.alert(
                    label=sanitized,
                    description="New project",
                    icon="folder-plus",
                    color="warning",
                    size="sm",
                )
        else:
            st.caption("No project — using temp directories")

        # Recent projects browser
        if find_projects:
            projects = find_projects(WORKDIR)

            if projects:
                st.caption("**Recent projects:**")

                # Build project items for tree display
                project_items = []
                for proj in projects[:8]:
                    proj_name = proj["name"]
                    num_runs = proj["num_runs"]
                    tag = sac.Tag(f"{num_runs} runs", color="blue") if num_runs > 0 else None
                    project_items.append(sac.TreeItem(
                        label=proj_name.replace("_", " "),
                        icon="folder2",
                        tag=tag,
                    ))

                selected_project = sac.tree(
                    items=project_items,
                    icon="archive",
                    size="sm",
                    height=150 if len(projects) > 3 else None,
                    checkbox=False,
                    return_index=False,
                    key="project_tree",
                )

                if selected_project:
                    proj_name = selected_project[-1] if isinstance(selected_project, list) else selected_project
                    if proj_name != st.session_state["project_name"]:
                        st.session_state["project_name"] = proj_name
                        if set_preference:
                            set_preference("last_project", proj_name)

    # ===================== WORKING DIRECTORY SECTION =====================
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### Working Directory")

    # Build quick locations (only show ones that exist)
    quick_locations = []
    location_paths = {}

    home_path = Path.home()
    quick_locations.append(sac.SegmentedItem(label="Home", icon="house"))
    location_paths["Home"] = home_path

    desktop = _get_known_folder("Desktop")
    if desktop:
        quick_locations.append(sac.SegmentedItem(label="Desktop", icon="display"))
        location_paths["Desktop"] = desktop

    docs = _get_known_folder("Documents")
    if docs:
        quick_locations.append(sac.SegmentedItem(label="Documents", icon="file-earmark-text"))
        location_paths["Documents"] = docs

    downloads = _get_known_folder("Downloads")
    if downloads:
        quick_locations.append(sac.SegmentedItem(label="Downloads", icon="download"))
        location_paths["Downloads"] = downloads

    # Segmented control for quick locations
    with st.sidebar:
        selected_location = sac.segmented(
            items=quick_locations,
            size="sm",
            radius="lg",
            color="blue",
            use_container_width=True,
            key="quick_location",
        )

    # Update tree base when quick location changes
    if selected_location and selected_location in location_paths:
        new_base = str(location_paths[selected_location])
        if new_base != st.session_state["tree_base"]:
            st.session_state["tree_base"] = new_base

    # Current tree base path
    tree_base = Path(st.session_state["tree_base"])
    if not tree_base.exists():
        tree_base = Path.home()
        st.session_state["tree_base"] = str(tree_base)

    # Build folder tree
    tree_items = _build_folder_tree(tree_base, max_depth=3)

    # Show current selection and path input
    with st.sidebar:
        # Compact path display with edit capability
        current_path = st.text_input(
            "Path",
            value=st.session_state["workdir"],
            key="workdir_input",
            label_visibility="collapsed",
            placeholder="Enter or paste a path...",
        )
        if current_path != st.session_state["workdir"]:
            st.session_state["workdir"] = current_path
            # Update WORKDIR immediately
            WORKDIR = Path(current_path).expanduser()
            WORKDIR.mkdir(parents=True, exist_ok=True)

        # Folder tree browser
        if tree_items:
            st.caption(f"Browse from: **{tree_base.name}/**")
            selected_folders = sac.tree(
                items=tree_items,
                icon="folder2-open",
                open_all=False,
                show_line=True,
                size="sm",
                height=200,
                checkbox=False,
                return_index=False,
                key="folder_tree",
            )

            # When user selects a folder in the tree
            if selected_folders:
                # Reconstruct full path from selection
                selected_name = selected_folders[-1] if isinstance(selected_folders, list) else selected_folders

                # Search for the actual path (handles nested selections)
                def find_path(base: Path, name: str, depth: int = 3) -> Path | None:
                    if depth <= 0:
                        return None
                    try:
                        for item in base.iterdir():
                            if item.is_dir() and item.name == name:
                                return item
                            if item.is_dir() and not item.name.startswith("."):
                                found = find_path(item, name, depth - 1)
                                if found:
                                    return found
                    except (PermissionError, OSError):
                        pass
                    return None

                found_path = find_path(tree_base, selected_name)
                if found_path:
                    st.session_state["workdir"] = str(found_path)
                    WORKDIR = found_path
        else:
            st.info("No accessible subfolders")

        # Action buttons row
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📂 Use this", use_container_width=True, type="primary", key="use_tree_base"):
                st.session_state["workdir"] = st.session_state["tree_base"]
                WORKDIR = Path(st.session_state["tree_base"])

        with col2:
            if st.button("💾 Set default", use_container_width=True, key="save_default"):
                if set_preference:
                    set_preference("default_workdir", st.session_state["workdir"])
                    st.toast("Default directory saved!", icon="✅")

    # Final validation and status display
    typed_path = Path(st.session_state["workdir"]).expanduser()
    WORKDIR = typed_path
    WORKDIR.mkdir(parents=True, exist_ok=True)

    with st.sidebar:
        if not typed_path.exists():
            sac.alert(
                label="Folder will be created",
                description=typed_path.name,
                icon="folder-plus",
                color="warning",
                size="sm",
            )
        elif not typed_path.is_dir():
            sac.alert(
                label="Not a folder",
                description="Please select a directory",
                icon="exclamation-triangle",
                color="error",
                size="sm",
            )
        else:
            # Show if this is the saved default
            is_default = saved_workdir and str(typed_path) == saved_workdir
            desc = str(typed_path.parent)
            if is_default:
                desc += " ⭐"
            sac.alert(
                label=typed_path.name,
                description=desc,
                icon="folder-check",
                color="success",
                size="sm",
            )

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


def _run_cli_with_progress(args: List[str], progress_placeholder, status_placeholder) -> str:
    """
    Run the MarkShark CLI with real-time progress updates.
    Parses stderr for [info] and [ok] messages to update progress.
    
    Args:
        args: CLI arguments
        progress_placeholder: Streamlit placeholder for progress bar
        status_placeholder: Streamlit placeholder for status text
        
    Returns:
        Combined stdout/stderr output
    """
    import re
    
    cmds = [
        ["markshark"] + args,
        [sys.executable, "-m", "markshark.cli"] + args,
    ]
    
    last_err = None
    for cmd in cmds:
        try:
            # Start process with line-buffered output
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )
            
            output_lines = []
            pages_processed = 0
            total_pages = None
            
            # Read stderr line by line for progress
            while True:
                line = proc.stderr.readline()
                if not line and proc.poll() is not None:
                    break
                if line:
                    output_lines.append(line)
                    
                    # Parse progress info from stderr
                    # Look for: "[info] Processing X scan pages as Y student(s) × Z page(s)"
                    match = re.search(r'Processing (\d+) scan pages', line)
                    if match:
                        total_pages = int(match.group(1))
                        status_placeholder.text(f"Processing {total_pages} pages...")
                    
                    # Look for: "[info] Aligning scan page X to template page Y"
                    match = re.search(r'Aligning scan page (\d+)', line)
                    if match:
                        current_page = int(match.group(1))
                        if total_pages:
                            progress = current_page / total_pages
                            progress_placeholder.progress(progress, text=f"Aligning page {current_page}/{total_pages}")
                        else:
                            status_placeholder.text(f"Aligning page {current_page}...")
                    
                    # Look for: "[ok]" or "[error]" completion messages
                    if '[ok]' in line or '[error]' in line:
                        pages_processed += 1
                        if total_pages:
                            progress = pages_processed / total_pages
                            progress_placeholder.progress(progress, text=f"Completed {pages_processed}/{total_pages} pages")
            
            # Read any remaining stdout
            stdout_out = proc.stdout.read() if proc.stdout else ""
            output_lines.insert(0, stdout_out)
            
            proc.wait()
            
            out = "".join(output_lines)
            if proc.returncode != 0:
                last_err = RuntimeError(out.strip() or f"Non-zero exit: {proc.returncode}")
                continue
            
            # Final progress update
            if total_pages:
                progress_placeholder.progress(1.0, text=f"Completed all {total_pages} pages")
            
            return out
            
        except FileNotFoundError as e:
            last_err = e
            continue
        except Exception as e:
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

def _template_selector_with_archive(key_prefix: str, label: str = "Select a pre-defined bubble sheet template"):
    """
    Unified template selector that shows active templates and optionally archived ones.

    Args:
        key_prefix: Unique prefix for Streamlit widget keys
        label: Label for the selectbox

    Returns:
        Selected BubbleSheetTemplate or None
    """
    if template_manager is None:
        return None

    # Get active and archived templates
    active_templates = template_manager.scan_templates()
    archived_templates = template_manager.scan_archived_templates()

    template_choice = None

    if active_templates:
        # Add favorite indicators
        template_options = []
        for t in active_templates:
            is_fav = template_manager.is_favorite(t.template_id)
            prefix = "⭐ " if is_fav else ""
            template_options.append(f"{prefix}{str(t)}")

        template_names = ["(none selected)"] + template_options
        selected_name = st.selectbox(label, template_names, key=f"{key_prefix}_active_select")

        if selected_name != "(none selected)":
            # Remove star prefix if present
            display_str = selected_name.replace("⭐ ", "")
            # Find the selected template
            for t in active_templates:
                if str(t) == display_str:
                    template_choice = t
                    break

            if template_choice:
                st.success(f"Using template: **{template_choice.display_name}**")
                if template_choice.num_questions:
                    st.caption(f"Questions: {template_choice.num_questions} | Choices: {template_choice.num_choices or 'N/A'}")
    else:
        st.info("No active templates found. Check archived templates below or upload custom files.")

    # Show archived templates in an expander
    if archived_templates:
        with st.expander(f"📦 Archived Templates ({len(archived_templates)})", expanded=False):
            st.caption("These templates have been archived. Select one to use it or unarchive from Template Manager.")
            archived_names = ["(none selected)"] + [str(t) for t in archived_templates]
            selected_archived = st.selectbox("Select archived template", archived_names, key=f"{key_prefix}_archived_select")

            if selected_archived != "(none selected)":
                # Find the selected archived template
                for t in archived_templates:
                    if str(t) == selected_archived:
                        template_choice = t
                        st.info(f"Using archived template: **{t.display_name}**")
                        if t.num_questions:
                            st.caption(f"Questions: {t.num_questions} | Choices: {t.num_choices or 'N/A'}")
                        break

    return template_choice

# --------------------- Sidebar ---------------------
# image_url = "https://github.com/navarrew/markshark/blob/main/images/shark.png" 
# st.sidebar.image(image_url, caption="MarkShark Logo", use_column_width=True)
st.sidebar.title("MarkShark Beta")
page = st.sidebar.radio("Select below", [
    "0) Quick grade",
    "1) Align scans",
    "2) Score",
    "3) Report",
    "4) Map visualizer",
    "5) Template manager",
    "6) Help"
])

# Initialize / show working directory selector
_init_workdir()

# Initialize template manager (used by multiple pages)
template_manager = None
template_manager_error = None
if TemplateManager is not None:
    try:
        templates_dir = _dflt(TEMPLATE_DEFAULTS, "templates_dir", None)
        template_manager = TemplateManager(templates_dir)
    except Exception as e:
        import traceback
        template_manager_error = f"{str(e)}\n{traceback.format_exc()}"  # Store full error for display

# ===================== 0) QUICK GRADE (UNIFIED WORKFLOW) =====================
if page.startswith("0"):
    st.header("Quick Grade")
    st.markdown("""
    Select a **template**, upload your **scanned answer sheets** and **answer key**, and MarkShark will:
    1. Align the scans to the template
    2. Score them automatically using default parameters
    """)
    
    # Top-of-page controls
    top_col1, top_col2 = st.columns([1, 3])
    with top_col1:
        run_quick_grade = st.button("Run Quick Grade", type="primary")
    with top_col2:
        quick_status = st.empty()
    
    st.divider()
    
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Inputs")
        # Template selection
        template_choice = _template_selector_with_archive("quick_grade")
        if template_choice:
            st.caption("✓ Bubble grid alignment fallback enabled")

        custom_template_pdf = None
        custom_bublmap = None

        scans = _tempfile_from_uploader("Upload your scanned answer sheets (PDF)", "quick_scans", types=("pdf",))
        key_txt = _tempfile_from_uploader("Upload your answer key (TXT)", "quick_key", types=("txt",))
        
        
        # Custom upload option
        if template_choice is None:
            st.markdown("---")
            st.markdown("**Upload custom template files below:**")
            st.caption("*Only if you are not using a pre-defined template from the menu above*")

            custom_template_pdf = _tempfile_from_uploader("Master template PDF", "quick_template_pdf", types=("pdf",))
            custom_bublmap = _tempfile_from_uploader("Bubblemap YAML", "quick_bubblemap", types=("yaml", "yml"))
            if custom_bublmap:
                st.caption("✓ Bubble grid alignment fallback enabled")
    
    with colB:
        st.subheader("Output file options")
        
        # Output names
        out_csv_name = st.text_input("Results CSV name", value="quick_grade_results.csv")
        out_pdf_name = st.text_input("Annotated PDF name", value="quick_grade_annotated.pdf")
        dpi = st.number_input("Render DPI", min_value=72, max_value=600, value=int(_dflt(RENDER_DEFAULTS, "dpi", 150)), step=1,
                              help="150 DPI is usually sufficient for bubble sheets. Higher values are slower.")
        
        st.markdown("---")
        
        # Scoring options

        st.subheader("Adjust parameters")

        with st.expander("Scoring options", expanded=False):
            annotate_all = st.checkbox("Annotate all bubbles", value=True)
            label_density = st.checkbox("Show % fill labels", value=True)
            auto_thresh = st.checkbox("Auto-calibrate threshold", value=_dflt(SCORING_DEFAULTS, "auto_calibrate_thresh", True))
            verbose_thresh = st.checkbox("Verbose threshold calibration", value=True)
            
            min_fill = st.text_input("Min fill", value="", placeholder=str(_dflt(SCORING_DEFAULTS, "min_fill", "")))
            min_top2_diff = st.text_input("Min top2 difference", value="", placeholder=str(_dflt(SCORING_DEFAULTS, "min_top2_diff", "")))
            top2_ratio = st.text_input("Top2 ratio", value="", placeholder=str(_dflt(SCORING_DEFAULTS, "top2_ratio", "")))
            
            st.markdown("---")
            st.markdown("**Flagging & Review**")
            generate_review_pdf = st.checkbox("Generate review PDF (flagged pages only)", value=True,
                                              help="Creates a PDF containing only pages with blank or ambiguous answers for manual review")
            generate_flagged_csv = st.checkbox("Generate flagged items CSV", value=True,
                                               help="Creates a CSV listing all flagged items (blank/multi answers) with student info")
            include_stats = st.checkbox("Include basic statistics in CSV", value=True,
                                        help="Appends exam stats (mean, std, KR-20) and item stats (% correct, point-biserial) to the CSV. Requires answer key.")
        
        # Alignment options
        with st.expander("Alignment options", expanded=False):
            align_method = st.selectbox(
                "Alignment method",
                ["auto", "fast", "slow", "aruco"],
                index=0,
                help="auto: fast if template selected, slow otherwise. "
                     "fast: 72 DPI coarse + bubble grid (quick). "
                     "slow: full-res ORB (thorough). "
                     "aruco: ArUco markers only."
            )
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

            # Determine output directory based on project mode
            project_name = st.session_state.get("project_name", "").strip()
            use_project_mode = bool(project_name and sanitize_project_name and create_project_structure and create_run_directory)

            if use_project_mode:
                # Project mode: create structured directories
                sanitized_name = sanitize_project_name(project_name)
                project_dir = create_project_structure(base, sanitized_name)
                work_dir, run_label = create_run_directory(project_dir)

                # Save input files to project/input/ for reference
                input_dir = project_dir / "input"
                if scans:
                    (input_dir / f"scans_{run_label}.pdf").write_bytes(scans.read_bytes())
                if key_txt:
                    (input_dir / f"answer_key_{run_label}.txt").write_bytes(key_txt.read_bytes())
                if template_choice:
                    # Save template reference
                    (project_dir / "logs" / "template_used.txt").write_text(f"{template_choice.template_id}\n{template_choice.display_name}")

                quick_status.info(f"📁 Project: {project_name} | Run: {run_label}")
            else:
                # Temporary mode: use temp directories as before
                work_dir = Path(tempfile.mkdtemp(prefix="quick_grade_", dir=str(base)))

            # Prepare log file path (for project mode)
            log_file = None
            if use_project_mode:
                from datetime import datetime
                log_file = project_dir / "logs" / f"log_{run_label}.txt"
                log_entries = []
                log_entries.append(f"MarkShark Quick Grade Log")
                log_entries.append(f"{'=' * 50}")
                log_entries.append(f"Project: {project_name}")
                log_entries.append(f"Run: {run_label}")
                log_entries.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                log_entries.append(f"{'=' * 50}\n")

            try:
                # Step 1: Align (with bubblemap for bubble grid fallback)
                quick_status.info("Step 1/2: Aligning scans to template...")

                # In project mode, save aligned PDF to project/aligned/ directory
                if use_project_mode:
                    aligned_dir = project_dir / "aligned"
                    aligned_pdf = aligned_dir / f"aligned_scans_{run_label}.pdf"
                else:
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
                
                # NEW: Pass bubblemap for bubble grid alignment fallback
                if bublmap:
                    align_args += ["--bubblemap", str(bublmap)]
                
                # Show progress during alignment
                align_progress = st.progress(0, text="Starting alignment...")
                align_status = st.empty()
                
                try:
                    align_out = _run_cli_with_progress(align_args, align_progress, align_status)
                except Exception as e:
                    # Fallback to regular CLI if progress version fails
                    align_out = _run_cli(align_args)
                
                align_progress.progress(1.0, text="✓ Alignment complete")
                quick_status.success("✓ Alignment complete")

                # Save alignment log
                if log_file:
                    log_entries.append(f"\n{'='*50}")
                    log_entries.append(f"STEP 1: ALIGNMENT")
                    log_entries.append(f"{'='*50}")
                    log_entries.append(f"Command: {' '.join(align_args)}")
                    log_entries.append(f"\nOutput:")
                    log_entries.append(align_out or "(no output)")

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
                if min_top2_diff.strip():
                    score_args += ["--min-top2-diff", min_top2_diff.strip()]
                
                # Stats option
                if include_stats:
                    score_args += ["--include-stats"]
                else:
                    score_args += ["--no-include-stats"]
                
                # Flagging/review options
                review_pdf_path = None
                flagged_csv_path = None
                if generate_review_pdf:
                    review_pdf_path = work_dir / "for_review.pdf"
                    score_args += ["--review-pdf", str(review_pdf_path)]
                if generate_flagged_csv:
                    flagged_csv_path = work_dir / "flagged.csv"
                    score_args += ["--flagged-csv", str(flagged_csv_path)]
                
                with st.spinner("Scoring sheets..."):
                    score_out = _run_cli(score_args)
                    quick_status.success("✅ Quick Grade complete!")

                # Save scoring log
                if log_file:
                    log_entries.append(f"\n{'='*50}")
                    log_entries.append(f"STEP 2: SCORING")
                    log_entries.append(f"{'='*50}")
                    log_entries.append(f"Command: {' '.join(score_args)}")
                    log_entries.append(f"\nOutput:")
                    log_entries.append(score_out or "(no output)")

                    # Write complete log to file
                    log_file.write_text('\n'.join(log_entries))
                    st.info(f"📝 Processing log saved to: logs/{log_file.name}")

                # Display results
                st.success("Processing complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    _download_file_button("📄 Download Results CSV", out_csv)
                with col2:
                    _download_file_button("📑 Download Annotated PDF", out_pdf)
                with col3:
                    _download_file_button("📋 Download Aligned Scans", aligned_pdf)
                
                # Flagging downloads (if generated)
                if review_pdf_path or flagged_csv_path:
                    flag_col1, flag_col2 = st.columns(2)
                    with flag_col1:
                        if review_pdf_path and review_pdf_path.exists():
                            _download_file_button("🔍 Download Review PDF (flagged pages)", review_pdf_path)
                    with flag_col2:
                        if flagged_csv_path and flagged_csv_path.exists():
                            _download_file_button("⚠️ Download Flagged Items CSV", flagged_csv_path)
                
                # Stats info
                if include_stats and key_txt:
                    st.info("📊 Basic statistics appended to CSV (scroll to bottom)")
                
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
        st.subheader("Inputs")
        # Template selection - either from library or custom upload
        align_template_choice = _template_selector_with_archive("align", "Select a pre-defined bubblesheet template")
        if align_template_choice:
            st.caption("✓ Fast alignment mode available (bubblemap included)")

        template = None
        align_bublmap = None

        scans = _tempfile_from_uploader("Upload your raw student scans (PDF)", "align_scans", types=("pdf",))
        

        st.markdown("---")
        
        # Custom upload option (shown if no template selected or no templates available)
        if align_template_choice is None:
            st.markdown("**Upload custom bubblesheet files:**")
            st.caption("*Only if you are not using a pre-defined template from the menu above*")
            template = _tempfile_from_uploader("Template bubble sheet (PDF)", "align_template", types=("pdf",))
            
            align_bublmap = _tempfile_from_uploader("Bubblemap YAML (optional)", "align_bubblemap", types=("yaml", "yml"))
            st.caption("*A bubblemap enables 'fast' alignment mode*")
            if align_bublmap:
                st.success("✓ Fast alignment mode available")
        

    with colB:
        st.subheader("Output file options")
        out_pdf_name = st.text_input("Output aligned PDF name", value="aligned_scans.pdf")
        dpi = st.number_input("Render DPI", min_value=72, max_value=600, value=int(_dflt(RENDER_DEFAULTS, "dpi", 150)), step=1)

        st.markdown("---")
        st.subheader("Alignment parameters")

        method = st.selectbox(
            "Alignment method",
            ["auto", "fast", "slow", "aruco"],
            index=0,
            help="auto: fast if bubblemap provided, slow otherwise. "
                 "fast: 72 DPI coarse + bubble grid (quick, needs bubblemap). "
                 "slow: full-res ORB (thorough). "
                 "aruco: ArUco markers only."
        )
        st.markdown("---")
        st.markdown("**ArUco mark alignment parameters**")
        min_markers = st.number_input("Min ArUco markers to accept", min_value=0, max_value=32, value=int(_dflt(ALIGN_DEFAULTS, "min_aruco", 0)), step=1)
        dict_name = st.text_input("ArUco dictionary (if aruco)", value=str(_dflt(ALIGN_DEFAULTS, "dict_name", "DICT_4X4_50")))

        st.markdown("---")
        st.markdown("**Non-ArUco align parameters**")
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
        first_page = st.number_input("First page to align in your raw scans (0 = auto)", min_value=0, value=0, step=1)
        last_page = st.number_input("Last page to align in your raw scans (0 = auto)", min_value=0, value=0, step=1)

    if run_align_clicked:
        # Determine template and bubblemap to use
        if align_template_choice is not None:
            actual_template = align_template_choice.template_pdf_path
            actual_bublmap = align_template_choice.bubblemap_yaml_path
        else:
            actual_template = template
            actual_bublmap = align_bublmap

        if not scans:
            status.error("Please upload scans PDF.")
        elif actual_template is None:
            status.error("Please select a template or upload a template PDF.")
        else:
            base = WORKDIR or Path(os.getcwd())

            # Check if we're in project mode
            project_name = st.session_state.get("project_name", "").strip()
            use_project_mode = bool(project_name and sanitize_project_name and create_project_structure and create_run_directory)

            if use_project_mode:
                # Project mode: save to project/aligned/
                sanitized_name = sanitize_project_name(project_name)
                project_dir = create_project_structure(base, sanitized_name)
                run_num = get_next_run_number(project_dir) if get_next_run_number else 1
                aligned_dir = project_dir / "aligned"
                out_pdf = aligned_dir / f"aligned_scans_{run_num:03d}.pdf"

                # Save input scans for reference
                input_dir = project_dir / "input"
                if scans:
                    (input_dir / f"scans_{run_num:03d}.pdf").write_bytes(scans.read_bytes())
            else:
                # Temporary mode
                out_dir = Path(tempfile.mkdtemp(prefix="align_", dir=str(base)))
                out_pdf = out_dir / out_pdf_name
            args = [
                "align",
                str(scans),
                "--template", str(actual_template),
                "--out-pdf", str(out_pdf),
                "--dpi", str(int(dpi)),
                "--align-method", method,
                "--estimator-method", estimator_method,
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
            
            # Pass bubblemap for fast alignment mode
            if actual_bublmap:
                args += ["--bubblemap", str(actual_bublmap)]

            try:
                # Show progress during alignment
                align_progress = st.progress(0, text="Starting alignment...")
                align_status_text = st.empty()
                
                try:
                    out = _run_cli_with_progress(args, align_progress, align_status_text)
                except Exception:
                    # Fallback to regular CLI if progress version fails
                    out = _run_cli(args)
                
                align_progress.progress(1.0, text="✓ Alignment complete")
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
        st.subheader("Input Files")

        # Template/Bubblemap selection
        score_template_choice = _template_selector_with_archive("score", "Select template from dropdown menu")

        bublmap = None

        aligned = _tempfile_from_uploader("Aligned scans PDF", "score_pdf", types=("pdf",))
        key_txt = _tempfile_from_uploader("Key TXT (optional)", "score_key", types=("txt",))

        st.markdown("Upload bubblemap below if not using a predefined template")
        # Custom upload option
        if score_template_choice is None:
            bublmap = _tempfile_from_uploader("Bubblemap (YAML)", "score_cfg", types=("yaml","yml"))

        st.markdown("---")
        st.subheader("Scoring Parameters")
        st.caption("*Adjust these if scoring isn't working well*")
        auto_thresh = st.checkbox("Auto-calibrate threshold", value=_dflt(SCORING_DEFAULTS, "auto_calibrate_thresh", True), key="score_auto_thresh")
        verbose_thresh = st.checkbox("Verbose threshold calibration", value=True, key="score_verbose")

        min_fill = st.text_input("Minimum bubble fill (leave blank for default)", value="", placeholder=str(_dflt(SCORING_DEFAULTS, "min_fill", "")))
        st.caption("*The fraction of the bubble that must be filled to be considered filled.*")

        min_top2_diff = st.text_input("Minimum top2 difference (leave blank for default)", value="", placeholder=str(_dflt(SCORING_DEFAULTS, "min_top2_diff", "")))
        st.caption("*The minimum fill difference (in percentage points) between 1st and 2nd-most filled bubbles to not score as multi.*")

        top2_ratio = st.text_input("Top2 ratio (leave blank for default)", value="", placeholder=str(_dflt(SCORING_DEFAULTS, "top2_ratio", "")))
        st.caption("*The minimum fill ratio between 1st and 2nd-most filled bubbles for 'multiple' fills.*")
        st.markdown("---")
        
        
        
    with colB:
        st.subheader("Output file options")
        out_csv_name = st.text_input("Output results CSV", value="results.csv")
        scored_pdf_name = st.text_input("Annotated scored PDF filename", value="", placeholder=str(_dflt(SCORING_DEFAULTS, "out_pdf", "scored_scans.pdf")))
        annotate_all = st.checkbox("Annotate all cells", value=True)
        label_density = st.checkbox("Label density diagnostics", value=True)
        dpi = st.number_input("Render DPI", min_value=72, max_value=600, value=int(_dflt(RENDER_DEFAULTS, "dpi", 150)), step=1, key="score_dpi")
        
        st.markdown("---")
        st.markdown("**Flagging & Review**")
        generate_review_pdf = st.checkbox("Generate review PDF (flagged pages only)", value=True,
                                          help="Creates a separate PDF with only pages containing blank/multi answers for manual review")
        generate_flagged_csv = st.checkbox("Generate flagged items CSV", value=True,
                                           help="Creates a CSV listing all blank/multi answers with student ID and question number")
        st.markdown("---")
        st.markdown("**Statistics**")
        include_stats = st.checkbox("Include basic statistics in CSV", value=True,
                                    help="Appends exam stats (mean, std, KR-20) and item stats (% correct, point-biserial) to the CSV. Requires answer key.")
        out_ann_dir = st.text_input("Annotated png directory (optional)", value="", placeholder="Enter a folder name here for png files")
        st.caption("*You can save annotated output student scans as a set of images (png format) by entering a directory name here.*")
        

    if run_score_clicked:
        # Determine bubblemap to use
        if score_template_choice is not None:
            actual_bublmap = score_template_choice.bubblemap_yaml_path
        else:
            actual_bublmap = bublmap

        if not aligned:
            score_status.error("Please upload aligned PDF.")
        elif actual_bublmap is None:
            score_status.error("Please select a template or upload a bubblemap YAML.")
        else:
            base = WORKDIR or Path(os.getcwd())

            # Check if we're in project mode
            project_name = st.session_state.get("project_name", "").strip()
            use_project_mode = bool(project_name and sanitize_project_name and create_project_structure and create_run_directory)

            if use_project_mode:
                # Project mode: save to project/results/run_XXX/
                sanitized_name = sanitize_project_name(project_name)
                project_dir = create_project_structure(base, sanitized_name)
                out_dir, run_label = create_run_directory(project_dir)
                out_csv = out_dir / out_csv_name
            else:
                # Temporary mode
                out_dir = Path(tempfile.mkdtemp(prefix="score_", dir=str(base)))
                out_csv = out_dir / out_csv_name
            args = [
                "score",
                str(aligned),
                "--bublmap", str(actual_bublmap),
                "--out-csv", str(out_csv),
                "--dpi", str(int(dpi)),
            ]
            if key_txt:
                args += ["--key-txt", str(key_txt)]
            if scored_pdf_name.strip():
                args += ["--out-pdf", scored_pdf_name.strip()]
            if out_ann_dir.strip():
                args += ["--out-annotated-dir", str(out_dir / out_ann_dir.strip())]
            if annotate_all:
                args += ["--annotate-all-cells"]
            if label_density:
                args += ["--label-density"]
            if not auto_thresh:
                args += ["--no-auto-thresh"]
            if verbose_thresh:
                args += ["--verbose-thresh"]
            if min_fill.strip():
                args += ["--min-fill", min_fill.strip()]
            if top2_ratio.strip():
                args += ["--top2-ratio", top2_ratio.strip()]
            if min_top2_diff.strip():
                args += ["--min-top2-diff", min_top2_diff.strip()]
            
            # Stats option
            if include_stats:
                args += ["--include-stats"]
            else:
                args += ["--no-include-stats"]
            
            # Review/flagging options
            review_pdf_path = None
            flagged_csv_path = None
            if generate_review_pdf:
                review_pdf_path = out_dir / "for_review.pdf"
                args += ["--review-pdf", str(review_pdf_path)]
            if generate_flagged_csv:
                flagged_csv_path = out_dir / "flagged.csv"
                args += ["--flagged-csv", str(flagged_csv_path)]

            try:
                with st.spinner("Scoring via CLI..."):
                    out = _run_cli(args)
                score_status.success("Scoring finished.")
                st.code(out or "Done.", language="bash")

                _download_file_button("Download results.csv", out_csv)
                
                # Show stats note if included
                if include_stats and key_txt:
                    st.info("📊 Basic statistics appended to CSV (scroll to bottom)")
                
                # If scored PDF was created, offer download
                if scored_pdf_name.strip():
                    scored_pdf_path = out_dir / scored_pdf_name.strip()
                    if scored_pdf_path.exists():
                        _download_file_button("Download scored PDF", scored_pdf_path)
                
                # If review PDF was created, offer download
                if review_pdf_path and review_pdf_path.exists():
                    _download_file_button("📋 Download Review PDF (flagged pages)", review_pdf_path)
                
                # If flagged CSV was created, offer download
                if flagged_csv_path and flagged_csv_path.exists():
                    _download_file_button("⚠️ Download Flagged Items CSV", flagged_csv_path)
                
                # If annotated dir was created, offer zip download
                if out_ann_dir.strip():
                    ann_path = out_dir / out_ann_dir.strip()
                    if ann_path.exists() and ann_path.is_dir():
                        zip_bytes = _zip_dir_to_bytes(ann_path)
                        st.download_button("Download annotated PNGs (zip)", data=zip_bytes, file_name="annotated.zip")

            except Exception as e:
                score_status.error(f"Error during scoring: {e}")

# ===================== 3) REPORT =====================
elif page.startswith("3"):
    st.header("Generate Excel Report")
    st.markdown("""
    Create a comprehensive Excel report with:
    - **Summary tab**: Overall exam statistics and reliability
    - **Per-version tabs**: Student results with color-coded item analysis
    - **Roster matching** (optional): Identify absent students and ID mismatches
    - **Visual highlighting**: Incorrect answers in red, quality indicators in green/yellow/red
    """)

    top_col1, top_col2 = st.columns([1, 3])
    with top_col1:
        run_report_clicked = st.button("Generate Report", type="primary")
    with top_col2:
        report_status = st.empty()

    st.divider()

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Required Input")
        results_csv = _tempfile_from_uploader("Results CSV (from score)", "report_csv", types=("csv",))

        # NEW: Show hint about where report will be saved
        if results_csv and st.session_state.get("project_name", "").strip():
            st.caption("💡 Tip: Report will be saved to the most recent run folder in your project")

    with colB:
        st.subheader("Optional Roster")
        st.caption("Upload a class roster to check for absent students and ID mismatches")
        roster_csv = _tempfile_from_uploader(
            "Class Roster CSV (StudentID, LastName, FirstName)",
            "roster_csv",
            types=("csv",),
        )
        if roster_csv:
            st.info("✓ Roster uploaded - will check for absent students and ID mismatches")

    st.divider()

    with st.expander("ℹ️ Roster CSV Format", expanded=False):
        st.markdown("""
        Your roster CSV should have these columns (case-insensitive):

        **Required:**
        - `StudentID` / `ID` / `Student_ID` - student identifier
        - `LastName` / `Last` / `Surname` - student last name

        **Optional:**
        - `FirstName` / `First` - student first name

        **Example:**
        ```csv
        StudentID,LastName,FirstName
        1001,SMITH,JOHN
        1002,JONES,SARAH
        1003,GARCIA,MARIA
        ```

        The report will:
        - ✓ Flag students on exam but not on roster (orphan scans)
        - ✓ List students on roster but no exam found (absent)
        - ✓ Use fuzzy matching to suggest possible ID matches
        """)

    out_filename = st.text_input("Output Excel filename", value="exam_report.xlsx")

    if run_report_clicked:
        if not results_csv:
            report_status.error("Please upload a results CSV.")
        else:
            base = WORKDIR or Path(os.getcwd())

            # Check if we're in project mode
            project_name = st.session_state.get("project_name", "").strip()
            use_project_mode = bool(project_name and sanitize_project_name and create_project_structure and create_run_directory)

            # Try to find the run folder this CSV belongs to
            csv_is_from_run_folder = False
            run_label = None
            csv_run_dir = None

            if use_project_mode:
                # Look for run folders in the current project
                sanitized_name = sanitize_project_name(project_name)
                project_dir = base / sanitized_name

                if project_dir.exists():
                    results_dir = project_dir / "results"
                    if results_dir.exists():
                        # Find the most recent run folder with a results.csv
                        import re
                        run_pattern = re.compile(r'^run_(\d+)_')

                        run_folders_with_csv = []
                        for run_dir in results_dir.iterdir():
                            if run_dir.is_dir():
                                match = run_pattern.match(run_dir.name)
                                if match:
                                    csv_in_run = run_dir / "results.csv"
                                    if csv_in_run.exists():
                                        run_num = int(match.group(1))
                                        run_folders_with_csv.append((run_num, run_dir))

                        # Get the most recent (highest number) run folder
                        if run_folders_with_csv:
                            run_folders_with_csv.sort(key=lambda x: x[0], reverse=True)
                            csv_run_dir = run_folders_with_csv[0][1]
                            run_label = csv_run_dir.name

                if csv_run_dir:
                    # Found a likely run folder - save report there
                    csv_is_from_run_folder = True
                    out_path = csv_run_dir / out_filename
                    report_status.info(f"📁 Saving report to run folder: {run_label}")
                else:
                    # No run folder found - create new one
                    project_dir = create_project_structure(base, sanitized_name)
                    work_dir, run_label = create_run_directory(project_dir)
                    out_path = work_dir / out_filename
            else:
                # Temporary mode
                out_dir = Path(tempfile.mkdtemp(prefix="report_", dir=str(base)))
                out_path = out_dir / out_filename
                run_label = None

            args = [
                "report",
                str(results_csv),
                "--out-xlsx", str(out_path),
            ]

            if roster_csv:
                args.extend(["--roster", str(roster_csv)])

            # Add project metadata if in project mode
            if use_project_mode:
                args.extend(["--project-name", project_name])
                if run_label:
                    args.extend(["--run-label", run_label])

            try:
                with st.spinner("Generating Excel report..."):
                    out = _run_cli(args)
                report_status.success("✅ Report generated!")
                st.code(out or "Done.", language="bash")

                if out_path.exists():
                    _download_file_button(f"📥 Download {out_filename}", out_path)

                    # Show preview of what's in the report
                    with st.expander("📊 What's in the report?", expanded=True):
                        st.markdown("""
                        **Summary Tab:**
                        - Overall exam statistics (N students, mean, std dev, KR-20)
                        - Reliability interpretation with color coding
                        - Roster issues (if roster provided)

                        **Per-Version Tabs (Version A, Version B, etc.):**
                        - Student results with columns: LastName, FirstName, StudentID, **Issue**, correct, incorrect, blank, multi, percent, Version, Q1, Q2...
                        - **Issue column** flags: blank answers, multi-marked, ID mismatch, fuzzy match
                        - **Incorrect answers highlighted in light red**
                        - **KEY row** showing correct answers
                        - **% Correct** - Item difficulty
                        - **Point-Biserial** - Item discrimination (green ≥0.20, yellow 0.10-0.20, red <0.10)
                        - **Item Quality** - Visual summary (✓ Good / ⚠ Review / ✗ Problem)
                        """)

            except Exception as e:
                report_status.error(f"Error: {e}")

# ===================== 4) VISUALIZE =====================
elif page.startswith("4"):
    st.header("Bubblemap Visualizer")
    st.markdown("Overlay bubble zones on a template or aligned PDF to verify placement.")

    top_col1, top_col2 = st.columns([1, 3])
    with top_col1:
        run_viz_clicked = st.button("Visualize")
    with top_col2:
        viz_status = st.empty()

    st.divider()

    colA, colB = st.columns(2)
    with colA:
        viz_pdf = _tempfile_from_uploader("PDF (template or aligned page)", "viz_pdf", types=("pdf",))
        viz_bublmap = _tempfile_from_uploader("Bubblemap YAML", "viz_yaml", types=("yaml", "yml"))
    with colB:
        out_image = st.text_input("Output image name", value="bubblemap_overlay.png")
        viz_dpi = st.number_input("Render DPI", min_value=72, max_value=600, value=300, step=1, key="viz_dpi")

    if run_viz_clicked:
        if not viz_pdf or not viz_bublmap:
            viz_status.error("Please upload PDF and bubblemap YAML.")
        else:
            base = WORKDIR or Path(os.getcwd())

            # Check if we're in project mode
            project_name = st.session_state.get("project_name", "").strip()
            use_project_mode = bool(project_name and sanitize_project_name and create_project_structure and create_run_directory)

            if use_project_mode:
                # Project mode: save to project/logs/
                sanitized_name = sanitize_project_name(project_name)
                project_dir = create_project_structure(base, sanitized_name)
                logs_dir = project_dir / "logs"
                out_path = logs_dir / out_image
            else:
                # Temporary mode
                out_dir = Path(tempfile.mkdtemp(prefix="viz_", dir=str(base)))
                out_path = out_dir / out_image

            args = [
                "visualize",
                str(viz_pdf),
                "--bublmap", str(viz_bublmap),
                "--out-image", str(out_path),
                "--dpi", str(int(viz_dpi)),
            ]

            try:
                with st.spinner("Generating overlay..."):
                    out = _run_cli(args)
                viz_status.success("Visualization complete.")
                st.code(out or "Done.", language="bash")

                if out_path.exists():
                    st.image(str(out_path), caption="Bubblemap Overlay")
                    _download_file_button("Download overlay image", out_path)

            except Exception as e:
                viz_status.error(f"Error: {e}")

# ===================== 5) TEMPLATE MANAGER =====================
elif page.startswith("5"):
    st.header("Template Manager")
    st.markdown("Manage your bubble sheet templates. Each template needs a PDF and a bubblemap YAML file.")
    
    if template_manager is not None:
        try:
            st.info(f"📁 Templates directory: `{template_manager.templates_dir}`")
            st.caption("You can change this by setting the MARKSHARK_TEMPLATES_DIR environment variable or editing defaults.py")
        except Exception as e:
            st.error(f"Could not initialize template manager: {e}")
    else:
        st.error("Template manager not available. Please ensure markshark.template_manager is installed.")
    
    if template_manager:
        # Top controls
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔄 Refresh Template List"):
                template_manager._templates_cache = None
                template_manager._archived_templates_cache = None
                st.rerun()

        st.divider()

        # Display existing templates with management controls
        templates = template_manager.scan_templates(force_refresh=False)
        archived_templates = template_manager.scan_archived_templates(force_refresh=False)

        if templates:
            st.subheader(f"📚 Active Templates ({len(templates)})")
            st.caption("Templates shown in order below will appear in the same order in dropdown menus throughout the app.")

            for idx, template in enumerate(templates):
                is_fav = template_manager.is_favorite(template.template_id)
                fav_icon = "⭐" if is_fav else "☆"

                with st.expander(f"{fav_icon} {template.display_name}", expanded=False):
                    # Template info
                    info_col, actions_col = st.columns([2, 1])

                    with info_col:
                        st.markdown(f"**Template ID:** `{template.template_id}`")
                        if template.description:
                            st.markdown(f"**Description:** {template.description}")
                        if template.num_questions:
                            st.markdown(f"**Questions:** {template.num_questions}")
                        if template.num_choices:
                            st.markdown(f"**Choices per question:** {template.num_choices}")

                        st.markdown(f"**PDF:** `{template.template_pdf_path.name}`")
                        st.markdown(f"**YAML:** `{template.bubblemap_yaml_path.name}`")

                    with actions_col:
                        # Validate template
                        is_valid, errors = template_manager.validate_template(template)
                        if is_valid:
                            st.success("✅ Valid")
                        else:
                            st.error("❌ Invalid")
                            for error in errors:
                                st.caption(f"• {error}")

                    # Management controls
                    st.markdown("**Manage Template:**")
                    action_col1, action_col2, action_col3, action_col4, action_col5 = st.columns(5)

                    with action_col1:
                        if st.button(f"{'⭐ Unfav' if is_fav else '☆ Favorite'}", key=f"fav_{template.template_id}", use_container_width=True):
                            template_manager.toggle_favorite(template.template_id)
                            st.rerun()

                    with action_col2:
                        if st.button("⬆️ Up", key=f"up_{template.template_id}", disabled=(idx == 0), use_container_width=True):
                            template_manager.move_template_up(template.template_id)
                            st.rerun()

                    with action_col3:
                        if st.button("⬇️ Down", key=f"down_{template.template_id}", disabled=(idx == len(templates) - 1), use_container_width=True):
                            template_manager.move_template_down(template.template_id)
                            st.rerun()

                    with action_col4:
                        if st.button("📦 Archive", key=f"archive_{template.template_id}", use_container_width=True):
                            if template_manager.archive_template(template.template_id):
                                st.success(f"Archived {template.display_name}")
                                st.rerun()
                            else:
                                st.error("Failed to archive template")

                    with action_col5:
                        pass  # Reserved for future actions

                    # Show full paths
                    if st.checkbox("Show full paths", key=f"paths_{template.template_id}"):
                        st.code(str(template.template_pdf_path))
                        st.code(str(template.bubblemap_yaml_path))
        else:
            st.warning("No active templates found in the templates directory.")

        # Archived templates section
        if archived_templates:
            st.divider()
            st.subheader(f"📦 Archived Templates ({len(archived_templates)})")
            st.caption("These templates are hidden from dropdown menus but can be restored.")

            for template in archived_templates:
                with st.expander(f"📋 {template.display_name} (archived)", expanded=False):
                    # Template info
                    info_col, actions_col = st.columns([2, 1])

                    with info_col:
                        st.markdown(f"**Template ID:** `{template.template_id}`")
                        if template.description:
                            st.markdown(f"**Description:** {template.description}")
                        if template.num_questions:
                            st.markdown(f"**Questions:** {template.num_questions}")

                    with actions_col:
                        # Validate template
                        is_valid, errors = template_manager.validate_template(template)
                        if is_valid:
                            st.success("✅ Valid")
                        else:
                            st.error("❌ Invalid")

                    # Unarchive button
                    if st.button(f"↩️ Restore to Active", key=f"unarchive_{template.template_id}"):
                        if template_manager.unarchive_template(template.template_id):
                            st.success(f"Restored {template.display_name}")
                            st.rerun()
                        else:
                            st.error("Failed to restore template")
        
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

**NEW: Bubble Grid Alignment Fallback**
When a bubblemap is provided during alignment, MarkShark can use the known bubble positions
to align scans even when ArUco markers are not present. This is especially useful for
legacy bubble sheets or templates from other sources.

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
