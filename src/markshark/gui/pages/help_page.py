"""
Help page - tabbed documentation with bundled Markdown help files.

Each tab loads a separate .md file from the assets/help/ directory.
Keeps documentation organised and avoids endless scrolling.
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextBrowser,
    QTabWidget,
)

from ..widgets import PageHeader

# Path to bundled help directory
_HELP_DIR = Path(__file__).parent.parent.parent / "assets" / "help"

# Ordered list of (tab_label, filename) for the help tabs
_HELP_TABS = [
    ("Getting Started", "getting_started.md"),
    ("Key & Roster Formats", "key_formats.md"),
    ("Alignment", "alignment.md"),
    ("Scoring", "scoring.md"),
    ("Templates", "templates.md"),
    ("LMS Integration", "lms_integration.md"),
    ("Reference", "reference.md"),
]


# ---------------------------------------------------------------------------
# Minimal Markdown-to-HTML converter (no external dependency)
# ---------------------------------------------------------------------------
def _md_to_html(md: str) -> str:
    """
    Convert a subset of Markdown to HTML suitable for QTextBrowser.

    Supports: headings (##), bold (**), italic (*), inline code (`),
    code blocks (```), unordered/ordered lists, horizontal rules,
    links, and tables.
    """
    import re

    lines = md.split("\n")
    html_lines: list[str] = []
    in_code_block = False
    in_list = False
    in_ol = False
    in_table = False

    for line in lines:
        # ── Code blocks ──
        if line.strip().startswith("```"):
            if in_code_block:
                html_lines.append("</pre>")
                in_code_block = False
            else:
                html_lines.append(
                    "<pre style=\"background-color: #f4f4f4; padding: 10px; "
                    "border-radius: 4px; font-family: 'Courier New', monospace; font-size: 12px;\">"
                )
                in_code_block = True
            continue

        if in_code_block:
            # Escape HTML inside code blocks
            escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_lines.append(escaped)
            continue

        stripped = line.strip()

        # ── Horizontal rule ──
        if stripped in ("---", "***", "___"):
            _close_lists(html_lines, in_list, in_ol, in_table)
            in_list = in_ol = in_table = False
            html_lines.append("<hr>")
            continue

        # ── Table rows ──
        if "|" in stripped and stripped.startswith("|"):
            # Skip separator rows like |---|---|
            if re.match(r"^\|[\s\-:|]+\|$", stripped):
                continue
            # Parse cells
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if not in_table:
                html_lines.append(
                    '<table style="border-collapse: collapse; margin: 8px 0;" '
                    'cellpadding="6" cellspacing="0">'
                )
                # First row is header
                html_lines.append("<tr>")
                for cell in cells:
                    html_lines.append(
                        f'<th style="border: 1px solid #ccc; padding: 6px 12px; '
                        f'background-color: #f0f0f0;">{_inline(cell)}</th>'
                    )
                html_lines.append("</tr>")
                in_table = True
            else:
                html_lines.append("<tr>")
                for cell in cells:
                    html_lines.append(
                        f'<td style="border: 1px solid #ccc; padding: 6px 12px;">'
                        f"{_inline(cell)}</td>"
                    )
                html_lines.append("</tr>")
            continue

        if in_table and not ("|" in stripped and stripped.startswith("|")):
            html_lines.append("</table>")
            in_table = False

        # ── Headings ──
        heading_match = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if heading_match:
            _close_lists(html_lines, in_list, in_ol, False)
            in_list = in_ol = False
            level = len(heading_match.group(1))
            text = heading_match.group(2)
            sizes = {1: "22px", 2: "18px", 3: "15px", 4: "13px", 5: "12px", 6: "11px"}
            sz = sizes.get(level, "13px")
            margin = "16px 0 8px 0" if level <= 2 else "12px 0 6px 0"
            html_lines.append(
                f'<h{level} style="font-size: {sz}; margin: {margin};">'
                f"{_inline(text)}</h{level}>"
            )
            continue

        # ── Unordered list ──
        ul_match = re.match(r"^[-*]\s+(.*)", stripped)
        if ul_match:
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{_inline(ul_match.group(1))}</li>")
            continue

        # ── Ordered list ──
        ol_match = re.match(r"^\d+\.\s+(.*)", stripped)
        if ol_match:
            if not in_ol:
                html_lines.append("<ol>")
                in_ol = True
            html_lines.append(f"<li>{_inline(ol_match.group(1))}</li>")
            continue

        # Close lists if we hit a non-list line
        if in_list:
            html_lines.append("</ul>")
            in_list = False
        if in_ol:
            html_lines.append("</ol>")
            in_ol = False

        # ── Empty line ──
        if not stripped:
            html_lines.append("<br>")
            continue

        # ── Paragraph ──
        html_lines.append(f"<p>{_inline(stripped)}</p>")

    # Close any open blocks
    if in_code_block:
        html_lines.append("</pre>")
    _close_lists(html_lines, in_list, in_ol, in_table)

    return "\n".join(html_lines)


def _close_lists(lines, in_list, in_ol, in_table):
    """Close any open list or table tags."""
    if in_list:
        lines.append("</ul>")
    if in_ol:
        lines.append("</ol>")
    if in_table:
        lines.append("</table>")


def _inline(text: str) -> str:
    """Process inline Markdown: bold, italic, code, links."""
    import re

    # Escape HTML (but preserve intentional tags we generate)
    text = text.replace("<", "&lt;").replace(">", "&gt;")

    # Inline code
    text = re.sub(r"`([^`]+)`", r"<code style=\"background-color: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-family: 'Courier New', monospace;\">\1</code>", text)

    # Bold + italic
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", text)
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Italic (only single asterisk, not inside words)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"<i>\1</i>", text)

    # Links: [text](url)
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<a href="\2" style="color: #1565C0;">\1</a>',
        text,
    )

    return text


def _render_markdown(md: str) -> str:
    """Convert markdown to styled HTML for a QTextBrowser."""
    html = _md_to_html(md)
    return (
        '<div style="font-family: Poppins, Roboto, Helvetica, Arial, sans-serif; '
        'max-width: 800px; margin: 0 auto; color: #333;">'
        f"{html}"
        "</div>"
    )


def _load_help_file(filename: str) -> str:
    """Load a markdown file from the help directory."""
    path = _HELP_DIR / filename
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            return f"# Error\n\nCould not load {filename}: {e}"
    return f"# {filename}\n\nHelp file not found."


_BROWSER_STYLE = (
    "QTextBrowser {"
    "  border: 1px solid #ccc;"
    "  padding: 16px;"
    "  font-size: 14px;"
    "  line-height: 1.5;"
    "  background-color: #ffffff;"
    "  color: #222222;"
    "}"
)


# ---------------------------------------------------------------------------
# Help page
# ---------------------------------------------------------------------------
class HelpPage(QWidget):
    """
    Help & Documentation page.

    Uses a QTabWidget with one tab per help topic. Each tab contains a
    QTextBrowser rendering a bundled Markdown file from assets/help/.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._browsers: dict[str, QTextBrowser] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = PageHeader(
            "MarkShark Help & Documentation",
            "User guide, reference, and keyboard shortcuts for MarkShark.",
        )
        layout.addWidget(header)

        # ── Toolbar row ──
        toolbar = QHBoxLayout()

        # Version info
        try:
            from markshark import __version__
            version = __version__
        except ImportError:
            version = "development"

        self.version_label = QLabel(f"Version: {version}")
        self.version_label.setStyleSheet("color: #666; font-size: 12px;")
        toolbar.addWidget(self.version_label)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # ── Tabbed help content ──
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        for tab_label, filename in _HELP_TABS:
            browser = QTextBrowser()
            browser.setOpenExternalLinks(True)
            browser.setStyleSheet(_BROWSER_STYLE)

            # Load and render the markdown
            md = _load_help_file(filename)
            browser.setHtml(_render_markdown(md))

            self.tabs.addTab(browser, tab_label)
            self._browsers[filename] = browser
