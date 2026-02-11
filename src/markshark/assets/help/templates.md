# Templates
A MarkShark template defines the layout of a bubble sheet: where the bubbles are, how many questions, what answer choices are available, and where student ID / name fields are located.
---
## Template Components
Each template consists of two files:
### Template PDF
The blank, unscanned master copy of the bubble sheet. This is the reference image that student scans are aligned against.
### Bubblemap YAML
A configuration file that tells MarkShark the exact position and size of every bubble grid on the template. It defines:
- **Answer zones** — Where the answer bubbles are (row/column grids)
- **ID zone** — Where the student ID bubbles are
- **Name zones** — Where first/last name bubbles are (if present)
- **Version zone** — Where the test version bubbles are (if present)
- **ArUco markers** — Positions of alignment markers
- **Styles** — Bubble shape (circle or oval), grid properties
- **Metadata** — Total questions, page count, page size
---
## Built-In Templates
MarkShark ships with several templates for common bubble sheet formats. Use the **Template Manager** page to browse, preview, and favourite templates.
---
## Bubblemap YAML Structure
A bubblemap has this general structure:
```
metadata:
  name: "My Template"
  total_questions: 80
  pages: 1
  page_size_mm: [215.9, 279.4]
styles:
  answer_bubbles:
    bubble_shape: circle
    bubble_radius_pct: 0.005
page_1:
  answer_zones:
    - name: "Questions 1-40"
      style: answer_bubbles
      labels: "ABCDE"
      numrows: 40
      numcols: 5
      top_left_mm: [22.5, 55.0]
      bottom_right_mm: [95.0, 265.0]
  id_zone:
    labels: "0123456789"
    numrows: 10
    numcols: 10
    top_left_mm: [110.0, 55.0]
    bottom_right_mm: [195.0, 120.0]
  version_zone:
    labels: "ABCD"
    numrows: 1
    numcols: 4
    top_left_mm: [110.0, 130.0]
    bottom_right_mm: [155.0, 140.0]
```
### Zone coordinates
Bubble positions can be specified in two ways:
**Millimetres** (recommended for new templates):
- `top_left_mm: [x, y]` — Top-left corner in mm from page origin
- `bottom_right_mm: [x, y]` — Bottom-right corner in mm
**Normalised fractions** (0.0 to 1.0):
- `top_left: [x, y]` — Top-left as fraction of page width/height
- `bottom_right: [x, y]` — Bottom-right as fraction
### Grid layout
- `numrows` — Number of rows (questions for answer zones, digits for ID)
- `numcols` — Number of columns (choices: A-E = 5, digits: 0-9 = 10)
- `labels` — The labels for each column ("ABCDE", "0123456789", etc.)
### Styles
Styles define shared properties across zones:
- `bubble_shape` — `circle` or `oval`
- `bubble_radius_pct` — Bubble radius as a fraction of page width
- `extends` — Inherit properties from another style
---
## Bubble Shapes
Templates can use either circular or oval bubbles:
- **Circle** — Equal width and height. Most common.
- **Oval** — Wider than tall (or vice versa). Some commercial bubble sheets use ovals.
The shape is defined in the bubblemap's `styles` section:
```
styles:
  answer_bubbles:
    bubble_shape: oval
  standard_bubbles:
    bubble_shape: circle
```
Answer zones reference styles with `style: answer_bubbles`.
---
## Multi-Page Templates
Templates can span multiple pages. Each page is defined separately:
```
metadata:
  pages: 2
  total_questions: 200
page_1:
  answer_zones:
    - name: "Questions 1-100"
      ...
page_2:
  answer_zones:
    - name: "Questions 101-200"
      ...
```
The student ID and version zones are typically only on page 1.
---
## Verifying Templates
Use the **Map Viewer** to overlay the bubblemap grid onto the template PDF. This shows exactly where MarkShark thinks the bubbles are. Check that:
- Grid circles/ovals sit centred on the printed bubbles
- Row and column counts are correct
- ID, name, and version zones are in the right positions
- ArUco marker positions match the printed markers
---
## Creating Custom Templates
To create a new template:
1. **Design** the bubble sheet in your preferred layout tool and export as PDF.
2. **Add ArUco markers** at known positions (at least 4 recommended).
3. **Create a bubblemap YAML** with the zone coordinates. Use millimetre coordinates for precision.
4. **Place both files** in a template directory under MarkShark's templates folder.
5. **Verify** with the Map Viewer before using in production.
The template directory should contain:
```
my_template/
  template.pdf
  bubblemap.yaml
```
