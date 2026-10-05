# Sheet Designer — Integration Plan

> **Status**: Planning (2026-03-04)
> **Goal**: Add a GUI page to MarkShark that lets users create and modify bubble sheets, producing both a printable PDF and a MarkShark-compatible bubblemap YAML.

---

## Overview

Integrate Bubblefish's sheet-generation engine into MarkShark as a new GUI page ("Sheet Designer"). Users configure their bubble sheet via form controls, see a live preview, and save the result as a new template (PDF + YAML) that MarkShark's `TemplateManager` picks up automatically.

The key architectural insight: **connect at the YAML boundary**. Bubblefish generates PDF + YAML → saves into a template folder → MarkShark's existing `TemplateManager.scan_templates()` discovers it. No data model unification needed.

---

## What Bubblefish Provides

Bubblefish (`~/Documents/GitHub/bubblefish/`) is a bubble sheet generator with:

- **SheetBuilder** — fluent API for defining sheets (page size, bubble size/shape, name fields, ID fields, answer grids, version fields, empty zones, output zones)
- **BuiltSheet** — immutable result with `.save_pdf()`, `.save_yaml()`, `.save_zip()`
- **Component model** — `NameField`, `IDField`, `VersionField`, `AnswerGrid`, `EmptyZone`, `OutputZone`
- **Enums/config** — `PageSize`, `BubbleShape`, `BubbleSize`, `ChoiceFormat`, `Spacing`, `FontFamily`, etc.
- **PDF renderer** — ReportLab-based, draws bubbles + ArUco markers + labels
- **YAML export** — outputs MarkShark v3 schema (mm-based coordinates)
- **YAML → PDF** — can reconstruct a PDF from any v3 YAML (round-trip capable)

### Bubblefish modules to bring in (~2,500 lines core)

| Module | Lines | Role |
|---|---|---|
| `builder.py` | ~350 | SheetBuilder fluent API |
| `components.py` | ~425 | Component classes (NameField, AnswerGrid, etc.) |
| `config.py` | ~330 | SheetConfig dataclass, enums, constants |
| `pdf_renderer.py` | ~400 | ReportLab → PDF |
| `yaml_export.py` | ~200 | BuiltSheet → YAML |
| `yaml_to_pdf.py` | ~200 | YAML → PDF (reconstruct from YAML) |
| `schema.py` | ~200 | YAML validation |
| `styles.py` | ~200 | Style resolution & unit conversion |
| `layout_calculator.py` | ~150 | Layout math |
| `presets/__init__.py` | ~100 | Built-in starter templates |

**Skip**: `cli.py`, `web/app.py` (FastAPI UI) — not needed in MarkShark.

---

## New Dependency

**ReportLab** (`reportlab>=4.0`) — PDF generation from scratch. Cross-platform, stable, widely used. Nothing in MarkShark's current stack (fitz, openpyxl, etc.) can create complex PDF layouts.

**Pillow is NOT needed** — Bubblefish uses it for PNG preview, but MarkShark already uses PyMuPDF (fitz) for PDF → image rendering. Replace Bubblefish's `preview_png()` with fitz render → QPixmap.

---

## Architecture

### Where the code lives

```
src/markshark/
├── tools/
│   └── sheet_builder/          # Bubblefish core (new sub-package)
│       ├── __init__.py         # Public API: SheetBuilder, BuiltSheet, presets
│       ├── builder.py
│       ├── components.py
│       ├── config.py
│       ├── pdf_renderer.py
│       ├── yaml_export.py
│       ├── yaml_to_pdf.py
│       ├── schema.py
│       ├── styles.py
│       ├── layout_calculator.py
│       └── presets.py
└── gui/
    └── pages/
        └── sheet_designer.py   # New GUI page
```

### Integration boundary

```
SheetBuilder (form controls map to API calls)
     ↓
BuiltSheet
     ↓
.save_pdf()  →  templates/<name>/master_template.pdf
.save_yaml() →  templates/<name>/bubblemap.yaml
     ↓
TemplateManager.scan_templates() picks it up automatically
```

No need to unify Bubblefish's mm-based `BuiltSheet`/`BubbleCoord` model with MarkShark's normalized-coordinate `Bubblemap`/`GridLayout` model. They meet at the YAML file.

### Preview pipeline (replaces Pillow)

```
SheetBuilder → BuiltSheet → .save_pdf(temp) → fitz.open(temp) → page.get_pixmap() → QPixmap
```

MarkShark's `PDFPreview` widget already does the fitz → QPixmap step.

---

## GUI Page Design

### Page: "Sheet Designer"

**Location in sidebar**: Under Templates section, or as a sub-action from Template Manager.

**Layout concept** (two-panel):

```
┌──────────────────────────────────────────────────────────┐
│  Sheet Designer                                    [?]   │
├────────────────────────┬─────────────────────────────────┤
│  CONFIGURATION         │  PREVIEW                        │
│                        │                                 │
│  ── Page Setup ──      │  ┌─────────────────────────┐   │
│  Page size: [Letter▼]  │  │                         │   │
│  Margins:   [10] mm    │  │   Live PDF preview      │   │
│                        │  │   (re-renders on change) │   │
│  ── Bubble Style ──    │  │                         │   │
│  Shape: [Circle▼]      │  │                         │   │
│  Size:  [Medium▼]      │  │                         │   │
│  Shading: [✓]          │  │                         │   │
│                        │  │                         │   │
│  ── Student Info ──    │  │                         │   │
│  Last Name: [✓] 8 chars│  │                         │   │
│  First Name: [✓] 6 ch  │  │                         │   │
│  Student ID: [✓] 6 dig │  │                         │   │
│  Version:   [✓] A-D    │  │                         │   │
│                        │  │                         │   │
│  ── Answer Grid ──     │  │                         │   │
│  Questions: [50]       │  │                         │   │
│  Choices:  [A-E ▼]     │  └─────────────────────────┘   │
│  Columns:  [2  ▼]      │                                 │
│                        │  [◀ Page 1 of 1 ▶]             │
│  ── Output Zone ──     │                                 │
│  Score area: [✓]       │                                 │
│                        │                                 │
│  ── Preset ──          │                                 │
│  [Load Preset ▼]       │                                 │
│                        │                                 │
│  [Generate & Save]     │  Template name: [___________]  │
├────────────────────────┴─────────────────────────────────┤
│  Status: Ready                                           │
└──────────────────────────────────────────────────────────┘
```

### Key form controls → SheetBuilder mapping

| Control | SheetBuilder method |
|---|---|
| Page size dropdown | `.page_size("letter"/"a4")` |
| Bubble shape dropdown | `.bubble_shape("circle"/"oval")` |
| Bubble size dropdown | `.bubble_size("small"/"medium"/"large")` |
| Alternating shading checkbox | `.alternating_shading(True/False)` |
| Last Name checkbox + char count | `.add_name_field("lastname", rows=N)` |
| First Name checkbox + char count | `.add_name_field("firstname", rows=N)` |
| Student ID checkbox + digit count | `.add_student_id(digits=N)` |
| Version checkbox + format | `.add_version(choices="A-D")` |
| Question count spinbox | `.add_answers(start=1, end=N, ...)` |
| Choice format dropdown | `.add_answers(..., choices="A-E")` |
| Column count dropdown | `.add_answers(..., columns=2)` |
| Score area checkbox | `.add_output_zone()` |
| Load Preset dropdown | Call preset function, populate form |

### Workflow

1. User adjusts form controls (or picks a preset to start from)
2. On each change (debounced ~300ms), rebuild `SheetBuilder` → `BuiltSheet` → render preview
3. User clicks "Generate & Save", enters a template name
4. Saves PDF + YAML into `templates/<name>/` directory
5. `TemplateManager` auto-discovers it — appears in template dropdown everywhere

### Edit existing template (stretch goal)

For templates generated by Bubblefish (identifiable by YAML metadata), the designer could:
1. Parse the YAML back into form controls (Bubblefish's `yaml_to_pdf.py` already does YAML → BuiltSheet)
2. Let user modify and re-save
3. This is the "round-trip" capability

For hand-made templates (existing MarkShark templates), editing would be limited to regenerating the PDF from the YAML (`yaml_to_pdf()`), not modifying the layout.

---

## Implementation Steps

### Phase 1: Import Bubblefish core
- [ ] Copy core modules into `src/markshark/tools/sheet_builder/`
- [ ] Remove Pillow dependency from preview code (replace with fitz)
- [ ] Add `reportlab>=4.0` to `pyproject.toml`
- [ ] Verify imports work, run Bubblefish's existing tests
- [ ] Test on both Mac and Windows

### Phase 2: Basic GUI page
- [ ] Create `gui/pages/sheet_designer.py` following existing page pattern
- [ ] Left panel: form controls for page setup, bubble style, student info, answer grid
- [ ] Right panel: PDF preview using `PDFPreview` widget
- [ ] Wire form controls → `SheetBuilder` → preview pipeline
- [ ] Add presets dropdown

### Phase 3: Save & integrate with TemplateManager
- [ ] "Generate & Save" button creates template folder with PDF + YAML
- [ ] Template appears in TemplateManager and all template dropdowns
- [ ] Optional: generate `preview.png` for template browser thumbnail

### Phase 4: Edit existing (stretch)
- [ ] Detect Bubblefish-generated templates (YAML metadata marker)
- [ ] Parse YAML back into form controls for editing
- [ ] Re-save modified template

### Phase 5: Polish
- [ ] Add to sidebar navigation
- [ ] Link from Template Manager page ("Create New" button)
- [ ] Help documentation
- [ ] Update CLAUDE.md

---

## Risks & Considerations

1. **PyInstaller bundle size** — ReportLab adds ~15 MB. Test builds on both platforms.
2. **Preview performance** — Full PDF render on every form change could be slow. Debounce at 300ms, consider a "preview" button instead of live updates if too sluggish.
3. **Multi-page sheets** — Bubblefish supports multi-page (e.g., 200Q), but the preview widget currently shows one page. Add page navigation if multi-page templates are common.
4. **Template naming conflicts** — Need to handle case where user picks a name that already exists.
5. **ArUco marker IDs** — Both repos use DICT_4X4_50 with IDs [0,1,2,3]. Keep this fixed; don't expose as a user option.

---

## Reference: Bubblefish Source

**Repo**: `/Users/williamnavarre/Documents/GitHub/bubblefish/`

Key files to study when implementing:
- `src/bubblefish/core/builder.py` — SheetBuilder API (this is what the GUI wraps)
- `src/bubblefish/core/config.py` — all enums and SheetConfig (maps to form controls)
- `src/bubblefish/core/pdf_renderer.py` — ReportLab rendering (keep as-is)
- `src/bubblefish/core/yaml_export.py` — YAML output (keep as-is)
- `src/bubblefish/core/yaml_to_pdf.py` — YAML → PDF for edit/round-trip
- `src/bubblefish/presets/__init__.py` — starter templates
- `docs/SCHEMA.md` — YAML schema spec (v3.2)
