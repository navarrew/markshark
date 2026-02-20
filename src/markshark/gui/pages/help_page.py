"""
Help page - tabbed documentation with bundled Markdown help files.

Each tab loads a separate .md file from the assets/help/ directory.
Keeps documentation organised and avoids endless scrolling.

Images are supported via standard Markdown syntax: ![alt](img/filename.png)
Image files live in assets/help/img/ and are resolved by QTextBrowser's
searchPaths mechanism (set to the help directory).
"""

from pathlib import Path

from PySide6.QtCore import QUrl
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
    ("Templates", "templates.md"),
    ("Alignment", "alignment.md"),
    ("Scoring", "scoring.md"),
    ("File Formats", "key_formats.md"),
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
    code blocks (```), nested unordered/ordered lists, horizontal rules,
    links, images (![alt](path)), and tables.
    """
    import re

    lines = md.split("\n")
    html_lines: list[str] = []
    in_code_block = False
    in_table = False

    # ── List nesting state ──
    # Stack of ("ul"|"ol") tags tracking open list levels.
    # Each entry corresponds to one indent level.  When indentation
    # increases we push a new tag; when it decreases we pop and close.
    list_stack: list[str] = []

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
            _close_all_lists(html_lines, list_stack)
            if in_table:
                html_lines.append("</table>")
                in_table = False
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
        # Each heading gets an anchor ID so you can link to it with
        # [Section Name](#section-name) from elsewhere on the page.
        heading_match = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if heading_match:
            _close_all_lists(html_lines, list_stack)
            level = len(heading_match.group(1))
            text = heading_match.group(2)
            # Build a URL-friendly anchor ID: lowercase, spaces→hyphens,
            # strip non-alphanumeric (except hyphens).
            anchor_id = re.sub(r"[^a-z0-9\-]", "", text.lower().replace(" ", "-"))
            sizes = {1: "22px", 2: "18px", 3: "15px", 4: "13px", 5: "12px", 6: "11px"}
            sz = sizes.get(level, "13px")
            margin = "16px 0 8px 0" if level <= 2 else "12px 0 6px 0"
            html_lines.append(
                f'<a name="{anchor_id}"></a>'
                f'<h{level} style="font-size: {sz}; margin: {margin};">'
                f"{_inline(text)}</h{level}>"
            )
            continue

        # ── Lists (unordered and ordered, with nesting) ──
        # Match against the ORIGINAL line to detect leading whitespace.
        # Every 2 spaces of indentation = one nesting level.
        ul_match = re.match(r"^(\s*)([-*])\s+(.*)", line)
        ol_match = re.match(r"^(\s*)\d+\.\s+(.*)", line) if not ul_match else None

        if ul_match or ol_match:
            if ul_match:
                indent = len(ul_match.group(1))
                item_text = ul_match.group(3)
                tag = "ul"
            else:
                indent = len(ol_match.group(1))
                item_text = ol_match.group(2)
                tag = "ol"

            # Convert indent to a depth level (0-based).
            # 2 spaces per level; be tolerant of 3-4 space indents.
            depth = indent // 2

            if not list_stack:
                # Starting a brand-new list
                html_lines.append(f"<{tag}>")
                list_stack.append(tag)
            elif depth >= len(list_stack):
                # Going deeper — open a nested list inside the current
                # (still-open) <li>.  The nested <ul>/<ol> sits inside
                # the parent item, which is how HTML nesting works.
                html_lines.append(f"<{tag}>")
                list_stack.append(tag)
            else:
                # Same depth or shallower — close deeper levels first,
                # then close the previous <li> at the target depth.
                while len(list_stack) > depth + 1:
                    closed = list_stack.pop()
                    html_lines.append(f"</li></{closed}>")
                # Close the previous <li> at this depth
                html_lines.append("</li>")
                # If the tag type changed at this level, swap it
                if list_stack and list_stack[-1] != tag:
                    old = list_stack.pop()
                    html_lines.append(f"</{old}>")
                    html_lines.append(f"<{tag}>")
                    list_stack.append(tag)

            html_lines.append(f"<li>{_inline(item_text)}")
            # Don't close </li> yet — a nested list might follow
            continue

        # Non-list line: close any open lists
        if list_stack:
            _close_all_lists(html_lines, list_stack)

        # ── Empty line ──
        if not stripped:
            html_lines.append("<br>")
            continue

        # ── Paragraph ──
        html_lines.append(f"<p>{_inline(stripped)}</p>")

    # Close any open blocks
    if in_code_block:
        html_lines.append("</pre>")
    _close_all_lists(html_lines, list_stack)
    if in_table:
        html_lines.append("</table>")

    return "\n".join(html_lines)


def _close_all_lists(lines: list, stack: list):
    """Unwind the entire list nesting stack, closing each level.

    Emits ``</li></ul>`` or ``</li></ol>`` for every entry in the stack,
    innermost first.  Mutates *stack* in place (empties it).
    """
    while stack:
        tag = stack.pop()
        lines.append(f"</li></{tag}>")


def _inline(text: str) -> str:
    """Process inline Markdown: bold, italic, code, images, links."""
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

    # Images: ![alt](path) — must come BEFORE links because ![...]()
    # would otherwise partially match the [...]() link pattern.
    # QTextBrowser resolves relative src paths via its searchPaths,
    # which we set to the help directory (see _setup_ui).
    text = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)",
        r'<img src="\2" alt="\1" style="max-width: 100%;">',
        text,
    )

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
        from ..utils import get_app_version
        self.version_label = QLabel(f"Version: {get_app_version()}")
        self.version_label.setStyleSheet("color: #666; font-size: 12px;")
        toolbar.addWidget(self.version_label)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # ── Tabbed help content ──
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        for tab_label, filename in _HELP_TABS:
            browser = QTextBrowser()
            # Don't let Qt handle link clicks — we intercept them below
            # so we can route .md links to the correct tab.
            browser.setOpenExternalLinks(False)
            browser.setStyleSheet(_BROWSER_STYLE)
            # Tell QTextBrowser where to find relative image paths
            # (e.g. "img/screenshot.png" → assets/help/img/screenshot.png).
            browser.setSearchPaths([str(_HELP_DIR)])

            # Intercept all link clicks so we can handle:
            #   - Cross-page links: [text](scoring.md) → load in place
            #   - Cross-page + anchor: [text](scoring.md#troubleshooting)
            #   - Same-page anchors: [text](#section-name) → scroll
            #   - External URLs: https://... → open in system browser
            browser.anchorClicked.connect(self._on_link_clicked)

            # Load and render the markdown
            md = _load_help_file(filename)
            browser.setHtml(_render_markdown(md))

            self.tabs.addTab(browser, tab_label)
            self._browsers[filename] = browser

        # Reset a tab to its own content whenever the user clicks it.
        # Uses tabBarClicked (not currentChanged) so re-clicking the
        # already-active tab also resets — important after a cross-page
        # link loaded foreign content into this tab's browser.
        self.tabs.tabBarClicked.connect(self._on_tab_clicked)

    # ------------------------------------------------------------------
    # Tab reset — clicking a tab reloads its own help file
    # ------------------------------------------------------------------

    def _on_tab_clicked(self, index: int):
        """Reload the clicked tab's own markdown content.

        Cross-page links load foreign content into the current browser
        (see _on_link_clicked).  Clicking any tab button resets it to
        that tab's own help file — acting as a reliable "home" for each
        topic.
        """
        if 0 <= index < len(_HELP_TABS):
            _, filename = _HELP_TABS[index]
            browser = self._browsers.get(filename)
            if browser:
                md = _load_help_file(filename)
                browser.setHtml(_render_markdown(md))

    # ------------------------------------------------------------------
    # Link navigation — cross-page, same-page, and external
    # ------------------------------------------------------------------

    def _on_link_clicked(self, url: QUrl):
        """Route clicked links to the right destination.

        Markdown links in help files can point to:
          [text](#anchor)                  → scroll within current page
          [text](other_page.md)            → load that file in place
          [text](other_page.md#anchor)     → load in place + scroll
          [text](https://example.com)      → open in system browser

        Cross-page links load content into the *current* browser rather
        than switching tabs.  The user can click the tab button to reset
        back to that tab's own content (see _on_tab_clicked).
        """
        import webbrowser as _wb

        url_str = url.toString()

        # ── External URL → system browser ──
        if url_str.startswith(("http://", "https://")):
            _wb.open(url_str)
            return

        # ── Same-page anchor (#section-name) ──
        fragment = url.fragment()
        path_part = url.path()  # e.g. "scoring.md" or ""

        if not path_part or path_part == "":
            # Pure anchor link like #troubleshooting
            if fragment:
                sender = self.sender()
                if isinstance(sender, QTextBrowser):
                    sender.scrollToAnchor(fragment)
            return

        # ── Cross-page link (other_page.md or other_page.md#anchor) ──
        # Strip leading slashes — QUrl may prepend one
        filename = path_part.lstrip("/")

        if filename in self._browsers:
            # Load the linked page's content INTO the current browser
            # (not switching tabs).  This keeps the user on the same tab
            # so they can click that tab button again to "go home" to
            # the tab's own content — see _on_tab_clicked().
            current_browser = self.tabs.currentWidget()
            if isinstance(current_browser, QTextBrowser):
                md = _load_help_file(filename)
                current_browser.setHtml(_render_markdown(md))
                if fragment:
                    current_browser.scrollToAnchor(fragment)
            return
