#!/usr/bin/env python3
"""
Quick-preview a MarkShark help markdown file in your web browser.

Uses the SAME converter that the MarkShark GUI uses, so what you see
in the browser is what teachers will see in the app.

Usage:
    python scripts/preview_help.py src/markshark/assets/help/getting_started.md
    python scripts/preview_help.py scoring.md          # looks in assets/help/
    python scripts/preview_help.py                     # previews ALL help files

The rendered HTML is written to a temp file and opened in your default
browser.  Edit the .md, re-run the script, and refresh the browser tab.
"""

import sys
import tempfile
import webbrowser
from pathlib import Path

# ── Locate the converter ──
# Add src/ to the import path so we can import the actual help_page
# converter functions — no duplication, guaranteed same rendering.
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
sys.path.insert(0, str(_SRC))

from markshark.gui.pages.help_page import _md_to_html, _inline  # noqa: E402

# Default help directory
_HELP_DIR = _SRC / "markshark" / "assets" / "help"


def _resolve_images(html: str, base_dir: Path) -> str:
    """Embed images as base64 data URIs so they display in ALL browsers.

    Safari blocks file:// cross-directory resource loading entirely, and
    Chrome requires a permission prompt.  Data URIs sidestep both issues
    by inlining the image bytes directly into the HTML.  The resulting
    file is larger, but for a handful of help screenshots it's fine.
    """
    import base64
    import mimetypes
    import re

    def _embed_src(match):
        src = match.group(1)
        # Already a data URI or remote URL — leave it alone
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)

        # Resolve relative paths against the markdown file's directory
        img_path = (base_dir / src).resolve()
        if not img_path.exists():
            print(f"  ⚠ Image not found: {img_path}")
            return match.group(0)

        # Guess MIME type from extension (default to PNG)
        mime, _ = mimetypes.guess_type(str(img_path))
        if mime is None:
            mime = "image/png"

        # Read image bytes → base64 → data URI
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        return f'<img src="data:{mime};base64,{b64}"'

    return re.sub(r'<img src="([^"]+)"', _embed_src, html)


def _render_full_page(md_text: str, base_dir: Path, title: str = "Help Preview") -> str:
    """Wrap the converted markdown in a complete HTML page.

    Images are embedded as base64 data URIs so they display in any
    browser without file:// permission issues (same rendering as
    QTextBrowser with setSearchPaths in the real app).
    """
    body = _md_to_html(md_text)
    # Rewrite relative image paths to absolute file:// URLs
    body = _resolve_images(body, base_dir)
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                         Roboto, Helvetica, Arial, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 0 20px;
            color: #333;
            line-height: 1.6;
            font-size: 15px;
        }}
        h1 {{ border-bottom: 2px solid #eee; padding-bottom: 8px; }}
        h2 {{ border-bottom: 1px solid #eee; padding-bottom: 4px; }}
        a {{ color: #1565C0; }}
        img {{ max-width: 100%; }}
        hr {{ border: none; border-top: 1px solid #ddd; margin: 24px 0; }}
    </style>
</head>
<body>
{body}
</body>
</html>"""


def preview_file(md_path: Path):
    """Convert one .md file and open it in the browser."""
    if not md_path.exists():
        print(f"Error: {md_path} not found", file=sys.stderr)
        sys.exit(1)

    md_text = md_path.read_text(encoding="utf-8")
    # Use the .md file's parent as the base directory for images,
    # so relative paths like img/screenshot.png work correctly.
    html = _render_full_page(md_text, md_path.parent, title=md_path.stem)

    # Write to a temp file and open in browser
    tmp = tempfile.NamedTemporaryFile(
        suffix=".html", prefix=f"markshark_help_{md_path.stem}_",
        delete=False, mode="w", encoding="utf-8",
    )
    tmp.write(html)
    tmp.close()
    print(f"Preview: {md_path.name}  →  {tmp.name}")
    webbrowser.open(f"file://{tmp.name}")


def preview_all():
    """Preview all help files as a single combined page."""
    if not _HELP_DIR.exists():
        print(f"Error: Help directory not found: {_HELP_DIR}", file=sys.stderr)
        sys.exit(1)

    md_files = sorted(_HELP_DIR.glob("*.md"))
    if not md_files:
        print("No .md files found in help directory", file=sys.stderr)
        sys.exit(1)

    # Combine all files with separators
    combined = []
    for md_file in md_files:
        combined.append(f"# {md_file.stem.replace('_', ' ').title()}\n")
        combined.append(md_file.read_text(encoding="utf-8"))
        combined.append("\n---\n")

    html = _render_full_page("\n".join(combined), _HELP_DIR, title="All Help Files")

    tmp = tempfile.NamedTemporaryFile(
        suffix=".html", prefix="markshark_help_all_",
        delete=False, mode="w", encoding="utf-8",
    )
    tmp.write(html)
    tmp.close()
    print(f"Preview: ALL help files  →  {tmp.name}")
    webbrowser.open(f"file://{tmp.name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # No argument — preview all help files
        preview_all()
    else:
        path = Path(sys.argv[1])
        # If just a filename (no directory), look in the help directory
        if not path.parent.parts or str(path.parent) == ".":
            candidate = _HELP_DIR / path
            if candidate.exists():
                path = candidate
        preview_file(path)
