
#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import csv
import os
import sys
import random
from typing import Dict, Any, List, Tuple, Optional

import yaml
from PIL import Image, ImageDraw

try:
    import img2pdf
    _HAS_IMG2PDF = True
except Exception:
    _HAS_IMG2PDF = False


r""""
bubble_synth.py

Synthesize filled bubbles from YAML + CSV and overlay them onto a blank bubble sheet. 
This allows you to invent data from imaginary students and see if your pipeline will
work with your bubblesheets and scanner.

Geometry:
  Each layout uses TL/BR as *bubble centers*:
    x_topleft, y_topleft   (normalized 0..1)
    x_bottomright, y_bottomright
  Interpolate centers over:
    questions = rows  (vertical)
    choices   = cols  (horizontal)
  Place one mark per column for ID/First/Last (selection_axis: "col"),
  and one mark per row for Version/Answers (selection_axis: "row").

CSV headers are normalized (BOM, spaces, hyphens, case):
  id / student_id / student-id / student number / student-number / studentid
  first_name / firstname / first
  last_name  / lastname  / last / surname
  version / test_version

Key file:
  --key-txt path supports either:
    - single-line key "ABCD..." (default for all records), or
    - versioned blocks:
        [A]
        ABCD...
        [B]
        BADC...
  Otherwise use --random-answers with --random-seed.

Defaults:
  - Composites onto --template by default (no separate overlays written)
  - If you want overlays too, pass --save-overlays
  - If you want a single PDF of all composited pages, pass --out-pdf somefile.pdf

Usage (common):
  python bubble_synth.py \
    --config 64Q-config.yaml \
    --csv fake_students.csv \
    --template 64Q-template.png \
    --out-dir out \
    --key-txt key.txt \
    --out-pdf fake_bubbles.pdf

Random instead of key:
  python bubble_synth.py \
    --config 64Q-config.yaml \
    --csv fake_students.csv \
    --template 64Q-template.png \
    --out-dir out \
    --random-answers --random-seed 123 \
    --out-pdf fake_bubbles.pdf

Example (lossless PNG → compact Letter PDF):
python bubble_synth_patched.py \
  --config 64Q-config.yaml \
  --csv fake_students.csv \
  --template 64Q-template.png \
  --out-dir out \
  --key-txt key.txt \
  --out-pdf out/fake_bubbles.pdf \
  --dpi 300 --image-format png

Smaller files (JPEG):
python bubble_synth_patched.py \
  --config 64Q-config.yaml \
  --csv fake_students.csv \
  --template 64Q-template.png \
  --out-dir out \
  --random-answers --random-seed 123 \
  --out-pdf out/fake_bubbles.pdf \
  --dpi 300 --image-format jpg --jpeg-quality 85
"""
#---------------------- GLOBAL CONSTANT FOR BUBBLE COLOR ----------------------------

"""
The rgba tuple defines the bubble color:
	•	The first three numbers (128,128,128) are the RGB values for a neutral mid-grey.
	•	The fourth number 220 is the alpha (opacity) on a 0–255 scale.
	
You can adjust the bubbles to make fake light marks, etc.
"""

# BUBBLE_RGBA = (100,100,100,90)  #Soft, light-opacity dark grey
# BUBBLE_RGBA = (100, 100, 100, 200)
# BUBBLE_RGBA = (200,200,200,180)  #Lighter grey, semi-transparent
# BUBBLE_RGBA = (64,64,64,255)  #Darker grey, fully opaque
BUBBLE_RGBA = (0,0,0,255)  #solid black bubbles


#-------------------FUNCTIONS--------------------

def normalize_key(k: str) -> str:
    return k.lstrip("\ufeff").strip().lower().replace(" ", "_").replace("-", "_")

def normalize_val(v) -> str:
    return "" if v is None else str(v).strip()

def normalize_record(row: dict) -> dict:
    return {normalize_key(k): normalize_val(v) for k, v in row.items()}

def get_norm(d_norm: dict, aliases) -> str:
    for a in aliases:
        key = normalize_key(a)
        if key in d_norm and d_norm[key] != "":
            return d_norm[key]
    return ""

def grid_centers_axis_mode(x_tl, y_tl, x_br, y_br, questions, choices):
    centers = []
    q_den = max(1, questions - 1)
    c_den = max(1, choices - 1)
    for r in range(questions):
        v = r / q_den
        y = y_tl + (y_br - y_tl) * v
        for c in range(choices):
            u = c / c_den
            x = x_tl + (x_br - x_tl) * u
            centers.append((x, y))
    return centers

def centers_to_radius_px(centers_pct, img_w, img_h, radius_pct):
    r_px = max(1, int(round(radius_pct * img_w)))
    pts_px = []
    for (cxp, cyp) in centers_pct:
        cx = int(round(float(cxp) * img_w))
        cy = int(round(float(cyp) * img_h))
        cx = max(0, min(cx, img_w - 1))
        cy = max(0, min(cy, img_h - 1))
        pts_px.append((cx, cy))
    return pts_px, r_px

def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def page_size(cfg: Dict[str, Any], template_path: Optional[str]) -> Tuple[int, int]:
    if template_path and os.path.exists(template_path):
        im = Image.open(template_path)
        return im.size
    page = cfg.get("page", {}) or {}
    w = page.get("width_px"); h = page.get("height_px")
    if w and h:
        return int(w), int(h)
    return 2550, 3300

def coerce_labels(labels):
    if isinstance(labels, str):
        return list(labels)
    if isinstance(labels, list):
        return [str(x) for x in labels]
    raise TypeError("labels must be string or list")


def draw_circle(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, rgba=BUBBLE_RGBA):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgba)

def fill_layout_by_columns(overlay, layout, text) -> int:
    if not text:
        return 0
    W, H = overlay.size
    labels = coerce_labels(layout["labels"])
    q = int(layout["questions"]); c = int(layout["choices"])
    centers_px, r_px = centers_to_radius_px(
        grid_centers_axis_mode(layout["x_topleft"], layout["y_topleft"],
                               layout["x_bottomright"], layout["y_bottomright"], q, c),
        W, H, float(layout.get("radius_pct", 0.008))
    )
    def idx(rr, cc): return rr * c + cc
    draw = ImageDraw.Draw(overlay, "RGBA")
    cnt = 0
    text = str(text)[:c]
    if any(ch.isalpha() for ch in labels):
        text = text.upper()
    for col_idx, ch in enumerate(text):
        try:
            row_idx = labels.index(ch)
        except ValueError:
            continue
        cx, cy = centers_px[idx(row_idx, col_idx)]
        draw_circle(draw, cx, cy, r_px); cnt += 1
    return cnt

def fill_layout_by_rows(overlay, layout, value) -> int:
    if not value:
        return 0
    W, H = overlay.size
    labels = coerce_labels(layout["labels"])
    q = int(layout["questions"]); c = int(layout["choices"])
    centers_px, r_px = centers_to_radius_px(
        grid_centers_axis_mode(layout["x_topleft"], layout["y_topleft"],
                               layout["x_bottomright"], layout["y_bottomright"], q, c),
        W, H, float(layout.get("radius_pct", 0.008))
    )
    def idx(rr, cc): return rr * c + cc
    draw = ImageDraw.Draw(overlay, "RGBA")
    cnt = 0
    seq = "".join(ch for ch in str(value).strip().upper() if ch.isalnum())
    if len(seq) < q: seq = seq + " " * (q - len(seq))
    seq = seq[:q]
    for row_idx, ch in enumerate(seq):
        try:
            col_idx = labels.index(ch)
        except ValueError:
            continue
        cx, cy = centers_px[idx(row_idx, col_idx)]
        draw_circle(draw, cx, cy, r_px); cnt += 1
    return cnt

def load_key_txt(path: str):
    version_to_key, default_key = {}, None
    cur_ver, buf = None, []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines()]
    has_sections = any(ln.strip().startswith("[") and ln.strip().endswith("]") for ln in lines)
    if has_sections:
        for ln in lines:
            s = ln.strip()
            if not s: continue
            if s.startswith("[") and s.endswith("]"):
                if cur_ver and buf:
                    version_to_key[cur_ver] = "".join(buf).replace(" ", "")
                    buf = []
                cur_ver = s.strip("[]").upper()
            else:
                buf.append(s)
        if cur_ver and buf:
            version_to_key[cur_ver] = "".join(buf).replace(" ", "")
    else:
        for ln in lines:
            s = ln.strip()
            if s:
                default_key = s.replace(" ", "")
                break
    return version_to_key, default_key

def random_answers(q: int, labels) -> str:
    labels = coerce_labels(labels)
    return "".join(random.choice(labels) for _ in range(q))

def build_overlay_for_record(cfg, page_wh, rec_norm, answers_mode):
    W, H = page_wh
    overlay = Image.new("RGBA", (W, H), (0,0,0,0))
    id_val    = get_norm(rec_norm, ["student_id","student-id","student number","student-number","studentid","id"])
    first_val = get_norm(rec_norm, ["first_name","firstname","first"])
    last_val  = get_norm(rec_norm, ["last_name","lastname","last","surname"])
    version   = get_norm(rec_norm, ["version","test_version"])

    lay = cfg.get("id_layout")
    if lay and lay.get("selection_axis","col").lower() == "col":
        fill_layout_by_columns(overlay, lay, id_val)
    lay = cfg.get("first_name_layout")
    if lay and lay.get("selection_axis","col").lower() == "col":
        fill_layout_by_columns(overlay, lay, first_val)
    lay = cfg.get("last_name_layout")
    if lay and lay.get("selection_axis","col").lower() == "col":
        fill_layout_by_columns(overlay, lay, last_val)
    vlay = cfg.get("version_layout")
    if vlay and vlay.get("selection_axis","row").lower() == "row":
        fill_layout_by_rows(overlay, vlay, version)

    seq_from_key = None
    vmap = answers_mode.get("version_to_key") or {}
    dkey = answers_mode.get("default_key")
    if vmap or dkey:
        ver_up = (version or "").upper()
        seq_from_key = vmap.get(ver_up, dkey)

    for alayout in cfg.get("answer_layouts", []):
        if alayout.get("selection_axis","row").lower() != "row":
            continue
        if seq_from_key:
            fill_layout_by_rows(overlay, alayout, seq_from_key)
        elif answers_mode.get("random_on"):
            q = int(alayout["questions"]); labs = alayout["labels"]
            rseq = random_answers(q, labs)
            fill_layout_by_rows(overlay, alayout, rseq)
    return overlay

def composite_on_base(base_image_path: str, overlay: Image.Image, target_wh=None) -> Image.Image:
    base = Image.open(base_image_path).convert("RGBA")
    if target_wh and base.size != target_wh:
        base = base.resize(target_wh, Image.LANCZOS)
    elif base.size != overlay.size:
        base = base.resize(overlay.size, Image.LANCZOS)
    return Image.alpha_composite(base, overlay)

def write_pdf_img2pdf(image_paths: List[str], out_pdf: str):
    if not _HAS_IMG2PDF:
        raise SystemExit("img2pdf is not installed. Install with `pip install img2pdf`.")
    letter = (8.5 * 72.0, 11.0 * 72.0)
    layout = img2pdf.get_layout_fun(letter)
    with open(out_pdf, "wb") as f:
        f.write(img2pdf.convert(image_paths, layout_fun=layout))

def generate_from_csv(cfg_path, csv_path, out_dir,
                      template_path=None,
                      composite_only=True,
                      save_overlays=False,
                      out_composited_dir=None,
                      key_txt_path=None,
                      random_answers_on=False,
                      random_seed=42,
                      out_pdf=None,
                      image_format="png",
                      dpi=300,
                      jpeg_quality=85):
    os.makedirs(out_dir, exist_ok=True)
    if not composite_only and not save_overlays:
        save_overlays = True
    if composite_only and not template_path:
        raise SystemExit("--template is required when composite_only=True (the default).")

    if template_path:
        out_composited_dir = out_composited_dir or os.path.join(out_dir, "composited")
        os.makedirs(out_composited_dir, exist_ok=True)

    cfg = load_cfg(cfg_path)

    random.seed(random_seed)
    version_to_key, default_key = ({}, None)
    if key_txt_path:
        version_to_key, default_key = load_key_txt(key_txt_path)

    answers_mode = {
        "random_on": bool(random_answers_on),
        "version_to_key": version_to_key,
        "default_key": default_key,
    }

    written_composited = []
    written_overlays = []

    target_wh = (int(round(8.5 * dpi)), int(round(11 * dpi)))  # e.g., (2550, 3300)
    W, H = target_wh  # <— lock overlay canvas to Letter@dpi regardless of template

    base_rgba = None
    if template_path:
        base = Image.open(template_path).convert("RGBA")
        if base.size != target_wh:
            base = base.resize(target_wh, Image.LANCZOS)
        base_rgba = base  # keep in memory to avoid re-opening every row

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            rec = normalize_record(row)
            overlay = build_overlay_for_record(cfg, (W, H), rec, answers_mode)
            
            sid = get_norm(rec, ["student_id","student-id","student number","student-number","studentid","id"]) or f"{i:03d}"

            if save_overlays:
                overlay_path = os.path.join(out_dir, f"overlay_{i:03d}_{sid}.{image_format}")
                if image_format.lower() in ("jpg","jpeg"):
                    overlay.convert("RGB").save(overlay_path, quality=jpeg_quality, subsampling=2, optimize=True, dpi=(dpi, dpi))
                else:
                    overlay.save(overlay_path, dpi=(dpi, dpi), optimize=True, compress_level=9)
                written_overlays.append(overlay_path)

            if template_path:
                comp = Image.alpha_composite(base_rgba, overlay)
                comp_dir = out_composited_dir if out_composited_dir else out_dir
                comp_path = os.path.join(comp_dir, f"sheet_{i:03d}_{sid}.{image_format}")
                if image_format.lower() in ("jpg","jpeg"):
                    comp.convert("RGB").save(comp_path, quality=jpeg_quality, subsampling=2, optimize=True, dpi=(dpi, dpi))
                else:
                    comp.convert("RGB").save(comp_path, dpi=(dpi, dpi), optimize=True, compress_level=9)
                written_composited.append(comp_path)

            if i % 50 == 0:
                print(f"... processed {i} records")

    if out_pdf and written_composited:
        if not _HAS_IMG2PDF:
            raise SystemExit("Cannot write PDF because img2pdf is not installed. `pip install img2pdf`")
        write_pdf_img2pdf(written_composited, out_pdf)
        print(f"Wrote PDF: {out_pdf}")

    print(f"Done. Composite pages: {len(written_composited)}; Overlays: {len(written_overlays)}")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Synthesize and composite fake bubble sheets from YAML + CSV.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--template", required=False, default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-pdf", required=False, default=None)

    ap.add_argument("--composite-only", action="store_true", default=True,
                    help="(Default) Only write composited pages (requires --template).")
    ap.add_argument("--save-overlays", action="store_true", default=False,
                    help="Also save transparent overlay PNGs.")

    ap.add_argument("--key-txt", default=None)
    ap.add_argument("--random-answers", action="store_true")
    ap.add_argument("--random-seed", type=int, default=42)

    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--image-format", default="png", choices=["png","jpg","jpeg"])
    ap.add_argument("--jpeg-quality", type=int, default=85)

    args = ap.parse_args()

    composite_only = args.composite_only
    if args.template is None and composite_only:
        print("[warn] --template is missing; switching to overlay-only output.", file=sys.stderr)
        composite_only = False

    generate_from_csv(
        cfg_path=args.config,
        csv_path=args.csv,
        out_dir=args.out_dir,
        template_path=args.template,
        composite_only=composite_only,
        save_overlays=args.save_overlays,
        out_composited_dir=None,
        key_txt_path=args.key_txt,
        random_answers_on=args.random_answers,
        random_seed=args.random_seed,
        out_pdf=args.out_pdfs if hasattr(args, "out_pdfs") else args.out_pdf,
        image_format=args.image_format,
        dpi=args.dpi,
        jpeg_quality=args.jpeg_quality,
    )

if __name__ == "__main__":
    main()
