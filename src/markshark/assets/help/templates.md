# Templates

Every bubble sheet in MarkShark is actually two files working together: a **bubble sheet PDF** (the printable page students fill in) and a **bubblemap** (a small text file that tells MarkShark where every bubble is and what it means). Together, these two files make up a **template**.

You don't usually need to think about this — MarkShark's built-in templates come with both files already paired up. But it helps to understand the relationship, especially if you ever need to troubleshoot alignment or create a custom sheet.

## Table of Contents
- [Using Templates](#using-templates)
- [What the Bubblemap Defines](#what-the-bubblemap-defines)
- [The Template Manager](#the-template-manager)
- [Verifying a Template](#verifying-a-template)
- [Multi-Page Templates](#multi-page-templates)
- [Creating Custom Templates](#creating-custom-templates)
- [Bubblemap YAML Reference](#bubblemap-yaml-reference)

---

## Using Templates

MarkShark ships with several built-in templates for common bubble sheet layouts. When you open the Grader page and select a template from the dropdown, MarkShark loads both the PDF and the bubblemap behind the scenes. The PDF is used as a reference image to align student scans against, and the bubblemap tells the scoring engine exactly where to look for filled bubbles.

To print blank sheets for your students, go to the **Template Manager**, select a template, and click "Save PDF" to save a copy you can print. You can also preview any template directly in the Template Manager to see what the sheet looks like before printing.

---

## What the Bubblemap Defines

The bubblemap is a YAML text file that maps out every bubble zone on the sheet. It tells MarkShark:

- **Answer zones** — where the answer bubbles are arranged in a grid, how many rows (questions) and columns (choices like A through E), and what labels each column represents.
- **Student ID zone** — where the student fills in their ID number by bubbling individual digits.
- **Name zones** — where students bubble in their first and last name (if the sheet has these fields).
- **Version zone** — where the student indicates which version of the test they received (A, B, C, etc.). Your answer key's version labels must match these — see [Answer Keys and Rosters](key_formats.md#version-labels-must-match-the-bubble-sheet).
- **ArUco markers** — the positions of the small square alignment markers printed on the sheet. MarkShark uses these to correct for rotation and skew when aligning student scans.
- **Metadata** — total number of questions, number of pages, page size, and display name.
- **Bubble shape** — whether the bubbles are circles or ovals.

---

## The Template Manager

The **Template Manager** page (in the sidebar) lets you browse and organize your installed templates. From here you can:

- **Preview** any template — see the PDF and its details (number of questions, pages, bubble shape, zones).
- **Favourite** templates you use often — favourites appear first in the Grader's template dropdown.
- **Reorder** templates using the Move Up / Move Down buttons to control the order they appear in dropdowns.
- **Archive** templates you no longer need — they're moved to an archive folder and hidden from dropdowns, but not deleted. You can unarchive them later.
- **Save** copies of the PDF or bubblemap YAML file to your computer.
- **Open** the template's folder in Finder (Mac) or Explorer (Windows) to inspect or edit the files directly.

---

## Verifying a Template

Before using a new or unfamiliar template, it's worth checking that the bubblemap lines up correctly with the printed PDF. Use the **Bubblemap Utility** (in the sidebar under utilities) to overlay the bubblemap grid onto the template PDF. You should see:

- Circles or ovals sitting centred on each printed bubble
- The correct number of rows and columns in each zone
- ID, name, and version zones in the right positions
- ArUco marker outlines matching the printed markers

If anything looks off, the bubblemap coordinates need adjustment. See [Bubblemap YAML Structure](#bubblemap-yaml-structure) below for how the coordinate system works.

---

## Multi-Page Templates

Some templates span two or more pages — for example, a 200-question exam might use page 1 for questions 1–100 and page 2 for questions 101–200. MarkShark handles this automatically. The student ID and version zones are typically only on page 1, and MarkShark groups consecutive pages together when scoring (pages 1+2 = student 1, pages 3+4 = student 2, and so on).

When printing a multi-page template, make sure students receive all pages and that scans stay in the correct order.

---

## Creating Custom Templates

If the built-in templates don't fit your needs, you can create your own. This requires:

1. **Design the bubble sheet** in your preferred layout tool (Word, InDesign, etc.) and export it as a PDF. Include ArUco alignment markers — at least four, placed near the corners. These small printed squares help MarkShark correct for scan rotation and skew.
2. **Create a bubblemap YAML file** that describes where every zone is on the page. See [Bubblemap YAML Structure](#bubblemap-yaml-structure) below for the format. Use millimetre coordinates for precision.
3. **Place both files in a template folder.** Each template lives in its own folder inside MarkShark's templates directory:
```
my_template/
  template.pdf
  bubblemap.yaml
```
4. **Verify with the Bubblemap Utility** — overlay the grid onto your PDF to make sure everything lines up before using the template with real exams.

MarkShark will automatically discover the new template and show it in the Template Manager and Grader dropdowns.

---

## Bubblemap YAML Reference

This section is a complete reference for anyone editing or creating bubblemap files. Most teachers won't need this — it's here for template authors and troubleshooting.

A bubblemap YAML file has four top-level sections: `metadata`, `styles` (optional), `registration` (optional), and one or more page sections (`page_1`, `page_2`, etc.).

### metadata

The metadata section describes the template as a whole.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `display_name` | string | — | Template name shown in the UI (required) |
| `description` | string | — | Longer description of the template |
| `pages` | int | 1 | Number of pages in the bubble sheet |
| `total_questions` | int | — | Total questions across all pages |
| `page_size` | string or dict | `"letter"` | Page size: `"letter"`, `"a4"`, `"legal"`, `"a3"`, or a dict with `width_mm` and `height_mm` for custom sizes |

**Standard page sizes (mm):**

| Name | Width | Height |
|------|-------|--------|
| `letter` | 215.9 | 279.4 |
| `a4` | 210.0 | 297.0 |
| `legal` | 215.9 | 355.6 |
| `a3` | 297.0 | 420.0 |

### styles (optional)

Styles define shared properties so you don't repeat the same settings for every zone. Each zone can reference a style with `style: style_name`, and the style's keys are merged into the zone. Zone-level keys always win if there's a conflict.

One style can inherit from another using `extends`:

```
styles:
  standard_bubbles:
    bubble_shape: circle
    bubble_diameter_mm: 3.8
  answer_bubbles:
    extends: standard_bubbles
    bubble_diameter_mm: 4.0
```

Common style keys include `bubble_shape` (`circle` or `oval`), `bubble_diameter_mm`, `bubble_radius_pct` (as a fraction of page width), and `bubble_stroke_width_mm`.

### registration (optional)

Configures how MarkShark aligns student scans against the template. If omitted, MarkShark uses sensible defaults.

```
registration:
  primary_method: aruco
  fallback_methods: [bubble_grid, orb_features]
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `primary_method` | string | `"aruco"` | Primary alignment method |
| `fallback_methods` | list | `["bubble_grid", "orb_features"]` | Methods to try if primary fails, in order |

**ArUco markers** (the small printed squares used for alignment):

```
  aruco:
    enabled: true
    dictionary: DICT_4X4_50
    marker_ids: [0, 1, 2, 3]
    size_mm: 6
    margin_mm: 6
    min_markers: 4
    ransac_threshold: 3.0
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Enable ArUco marker detection |
| `dictionary` | string | `"DICT_4X4_50"` | ArUco dictionary (OpenCV naming) |
| `marker_ids` | list | — | Which ArUco IDs are printed on the sheet (e.g. `[0, 1, 2, 3]`) |
| `size_mm` | float | — | Printed marker size in millimetres |
| `margin_mm` | float | — | Distance from page edge to marker centre in millimetres |
| `min_markers` | int | 4 | Minimum markers required for alignment |
| `ransac_threshold` | float | 3.0 | Error threshold in pixels for homography |

**Bubble grid alignment** (uses the printed bubble grids themselves for alignment):

```
  bubble_grid:
    enabled: true
    ransac_threshold: 5.0
    min_inliers: 30
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable bubble grid alignment |
| `hough_param1` | int | 50 | Canny edge detection threshold |
| `hough_param2` | int | 25 | Hough circle accumulator threshold |
| `ransac_threshold` | float | 5.0 | Error threshold in pixels |
| `min_inliers` | int | 30 | Minimum matching bubbles required |

**ORB feature alignment** (uses image keypoints for alignment):

```
  orb_features:
    enabled: true
    orb_nfeatures: 2000
    match_ratio: 0.75
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable ORB keypoint alignment |
| `orb_nfeatures` | int | 2000 | Maximum features to detect |
| `match_ratio` | float | 0.75 | Lowe's ratio threshold for matching |

---

### Page sections

Each page is defined under `page_1`, `page_2`, etc. A page section contains one or more **zones** — rectangular regions of the sheet that MarkShark reads during scoring.

#### Zone coordinates

Every zone needs coordinates that tell MarkShark where it sits on the page. There are two coordinate formats:

**Millimetres** (recommended for new templates):
- `x_mm`, `y_mm` — top-left corner in mm from the page origin
- `width_mm`, `height_mm` — size in mm
- `bubble_diameter_mm` — bubble size (default 4.0)

**Normalised fractions** (0.0 to 1.0, used by some older templates):
- `x_topleft`, `y_topleft` — top-left as a fraction of page width/height
- `x_bottomright`, `y_bottomright` — bottom-right as a fraction
- `radius_pct` — bubble radius as a fraction of page width

The coordinates define a bounding box. MarkShark divides this box into a grid based on `numrows` and `numcols` to find the centre of each bubble.

#### Common zone keys

These keys are shared by all bubble grid zones (answer zones, ID zone, name zones, version zone):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `numrows` | int | — | Number of rows in the grid (required) |
| `numcols` | int | — | Number of columns in the grid (required) |
| `labels` | string | auto | Label for each row or column — see auto-defaults below |
| `selection_axis` | string | `"row"` | `"row"` = one column selected per row (answer grids); `"col"` = one row selected per column (ID and name grids) |
| `bubble_shape` | string | `"circle"` | `"circle"` or `"oval"` |
| `style` | string | — | Name of a style from the `styles` section to inherit |
| (coordinates) | — | — | Either mm or normalised format (required) |

#### answer_zones

A list of one or more bubble grids for multiple-choice answers. Most sheets have one or two answer zones (left column and right column of questions).

```
page_1:
  answer_zones:
    - numrows: 40
      numcols: 5
      selection_axis: row
      labels: "ABCDE"
      style: answer_bubbles
      x_mm: 20
      y_mm: 55
      width_mm: 75
      height_mm: 210
    - numrows: 40
      numcols: 5
      selection_axis: row
      labels: "ABCDE"
      style: answer_bubbles
      x_mm: 110
      y_mm: 55
      width_mm: 75
      height_mm: 210
```

If `labels` is omitted, MarkShark auto-generates `"ABCDE..."` based on `numcols`.

#### id_zone (optional)

Bubble grid where students fill in their student ID number, one digit per column.

```
  id_zone:
    numrows: 10
    numcols: 8
    selection_axis: col
    x_mm: 110
    y_mm: 55
    width_mm: 40
    height_mm: 65
```

- `selection_axis` defaults to `"col"` (one row selected per column — each column is a digit position)
- `labels` defaults to `"0123456789"` if omitted

#### last_name_zone / first_name_zone (optional)

Bubble grids where students bubble in their name, one letter per column.

```
  last_name_zone:
    numrows: 27
    numcols: 14
    selection_axis: col
    x_mm: 82
    y_mm: 30
    width_mm: 82
    height_mm: 124
```

- `selection_axis` defaults to `"col"`
- `labels` defaults to `" ABCDEFGHIJKLMNOPQRSTUVWXYZ"` (space + 26 letters) if omitted — the leading space represents a blank/unused position

#### test_id_zone (optional)

Bubble grid for a numerical test ID (distinct from the version zone). Same structure as `id_zone`.

- `selection_axis` defaults to `"col"`
- `labels` defaults to `"0123456789"` if omitted

#### version_zone (optional)

Bubble grid where students indicate which version of the test they received.

```
  version_zone:
    numrows: 1
    numcols: 4
    selection_axis: row
    labels: "ABCD"
    x_mm: 110
    y_mm: 130
    width_mm: 45
    height_mm: 10
```

- `selection_axis` defaults to `"row"`
- `labels` defaults to `"ABCD..."` based on `numcols` if omitted
- The labels here determine what version identifiers MarkShark outputs during scoring — your answer key's `ver:` headers must match these labels

#### output_zone (optional)

A rectangular region (not a bubble grid) where MarkShark writes the student's name, score, and other annotation text onto the scored PDF.

```
  output_zone:
    x_mm: 20
    y_mm: 2
    width_mm: 148
    height_mm: 17
```

This zone has only coordinates — no `numrows`, `numcols`, or `labels`.

---

### Complete example

This is the actual bubblemap from the built-in MarkShark 200 Question Template — a two-page, four-column layout with name, ID, and version zones on page 1. Sections used only by the Bubblefish PDF generator (`fonts`, `text_zones`, `textbox_zones`, `line_zones`) are omitted here.

```
metadata:
  display_name: MarkShark 200 Question Template
  description: 200 questions, 5 choices (A-E)
  pages: 2
  total_questions: 200
  schema_version: 3
  page_size: letter

registration:
  primary_method: aruco
  aruco:
    enabled: true
    dictionary: DICT_4X4_50
    marker_ids: [0, 1, 2, 3]
    size_mm: 6
    margin_mm: 6

styles:
  standard_bubbles:
    bubble_shape: circle
    bubble_diameter_mm: 3.8
    bubble_stroke_width_mm: 0.25

  answer_bubbles:
    extends: standard_bubbles
    bubble_diameter_mm: 4

page_1:
  output_zone:
    x_mm: 20
    y_mm: 2.0
    width_mm: 148
    height_mm: 17

  last_name_zone:
    style: standard_bubbles
    numrows: 27
    numcols: 14
    selection_axis: col
    labels: ' ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    x_mm: 82
    y_mm: 30
    width_mm: 82.13
    height_mm: 123.89

  first_name_zone:
    style: standard_bubbles
    numrows: 27
    numcols: 5
    selection_axis: col
    labels: ' ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    x_mm: 176
    y_mm: 30
    width_mm: 25.26
    height_mm: 123.89

  id_zone:
    style: standard_bubbles
    numrows: 10
    numcols: 10
    selection_axis: col
    labels: 0123456789
    x_mm: 14
    y_mm: 111
    width_mm: 56
    height_mm: 43

  version_zone:
    style: standard_bubbles
    numrows: 1
    numcols: 10
    selection_axis: row
    labels: ABCDEFGHIJ
    x_mm: 14
    y_mm: 93
    width_mm: 56

  answer_zones:
  - style: answer_bubbles
    numrows: 15
    numcols: 5
    selection_axis: row
    labels: ABCDE
    x_mm: 20
    y_mm: 170
    width_mm: 30
    height_mm: 90
  - style: answer_bubbles
    numrows: 15
    numcols: 5
    selection_axis: row
    labels: ABCDE
    x_mm: 71
    y_mm: 170
    width_mm: 30
    height_mm: 90
  - style: answer_bubbles
    numrows: 15
    numcols: 5
    selection_axis: row
    labels: ABCDE
    x_mm: 122
    y_mm: 170
    width_mm: 30
    height_mm: 90
  - style: answer_bubbles
    numrows: 15
    numcols: 5
    selection_axis: row
    labels: ABCDE
    x_mm: 173
    y_mm: 170
    width_mm: 30
    height_mm: 90

page_2:
  answer_zones:
  - style: answer_bubbles
    numrows: 35
    numcols: 5
    selection_axis: row
    labels: ABCDE
    x_mm: 20
    y_mm: 20
    width_mm: 30
    height_mm: 240
  - style: answer_bubbles
    numrows: 35
    numcols: 5
    selection_axis: row
    labels: ABCDE
    x_mm: 71
    y_mm: 20
    width_mm: 30
    height_mm: 240
  - style: answer_bubbles
    numrows: 35
    numcols: 5
    selection_axis: row
    labels: ABCDE
    x_mm: 122
    y_mm: 20
    width_mm: 30
    height_mm: 240
  - style: answer_bubbles
    numrows: 35
    numcols: 5
    selection_axis: row
    labels: ABCDE
    x_mm: 173
    y_mm: 20
    width_mm: 30
    height_mm: 240
```
