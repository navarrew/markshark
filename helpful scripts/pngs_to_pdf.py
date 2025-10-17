#!/usr/bin/env python3
"""
Combine PNG files into a multi-page PDF.

Default: preserve each image at its native pixel size (Pillow backend).
Optional: --page-size/--margin uses ReportLab to fit images onto a uniform page.

Examples:
  # Basic (current folder -> out.pdf; sorts naturally)
  python pngs_to_pdf.py -i . -o out.pdf

  # Recursive, filter filename pattern, reverse sort
  python pngs_to_pdf.py -i scans -o exam.pdf --recursive --pattern "*.png" --sort name --reverse

  # Set output DPI metadata (affects printed size)
  python pngs_to_pdf.py -i scans -o exam.pdf --dpi 300

  # Force Letter pages with 0.5 inch margins, scale-to-fit (ReportLab)
  python pngs_to_pdf.py -i scans -o exam.pdf --page-size Letter --margin 0.5

Requires: Pillow
Optional: reportlab (only if you use --page-size/--margin)
"""

import argparse
import fnmatch
import os
import re
import sys
from pathlib import Path

from PIL import Image

# -------- Sorting helpers --------

_num_chunk_re = re.compile(r'(\d+)', re.UNICODE)

def natural_key(s: str):
    """Natural sort key (splits numbers so 2 < 10)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in _num_chunk_re.split(s)]

def list_pngs(input_dir: Path, pattern: str, recursive: bool):
    if recursive:
        files = [p for p in input_dir.rglob('*') if p.is_file() and fnmatch.fnmatch(p.name, pattern)]
    else:
        files = [p for p in input_dir.iterdir() if p.is_file() and fnmatch.fnmatch(p.name, pattern)]
    # keep only .png/.PNG by extension too (pattern may be broad)
    files = [p for p in files if p.suffix.lower() == '.png']
    return files

def sort_files(files, mode: str, reverse: bool):
    if mode == 'name':
        files.sort(key=lambda p: natural_key(p.name), reverse=reverse)
    elif mode == 'mtime':
        files.sort(key=lambda p: p.stat().st_mtime, reverse=reverse)
    elif mode == 'numeric':
        # Extract the largest number in filename; fallback to 0
        def file_num(p):
            nums = [int(n) for n in _num_chunk_re.findall(p.stem)]
            return max(nums) if nums else -1
        files.sort(key=file_num, reverse=reverse)
    else:
        files.sort(key=lambda p: natural_key(p.name), reverse=reverse)
    return files

# -------- Image prep --------

def open_as_rgb(path: Path, bg_color=(255, 255, 255)):
    """Open an image and ensure it's RGB (flatten RGBA onto white)."""
    im = Image.open(path)
    im.load()
    if im.mode in ('RGBA', 'LA'):
        bg = Image.new('RGB', im.size, bg_color)
        if im.mode == 'LA':
            # Convert LA -> RGBA first
            im = im.convert('RGBA')
        bg.paste(im, mask=im.split()[-1])  # use alpha channel as mask
        return bg
    elif im.mode == 'P':
        return im.convert('RGB')
    elif im.mode != 'RGB':
        return im.convert('RGB')
    return im

# -------- Backends --------

def save_with_pillow(images_rgb, out_path: Path, dpi: int | None):
    if not images_rgb:
        raise SystemExit("No images to save.")
    first, rest = images_rgb[0], images_rgb[1:]
    save_kwargs = {"save_all": True, "append_images": rest}
    if dpi:
        # Pillow writes DPI tuple into PDF metadata for sizing
        save_kwargs["resolution"] = dpi
    # Note: Pillow’s PDF plugin ignores per-page DPI; it’s metadata for display/print.
    first.save(out_path, "PDF", **save_kwargs)

def save_with_reportlab(image_paths, out_path: Path, page_size_name: str, margin_in: float):
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.units import inch
    except ImportError:
        raise SystemExit("ReportLab is required for --page-size/--margin. Install with: pip install reportlab")

    page_sizes = {
        "Letter": letter,
        "A4": A4,
    }
    if page_size_name not in page_sizes:
        raise SystemExit(f"Unsupported page size: {page_size_name}. Choose from: {', '.join(page_sizes)}")

    page_w, page_h = page_sizes[page_size_name]
    c = canvas.Canvas(str(out_path), pagesize=page_sizes[page_size_name])
    margin = margin_in * inch
    avail_w = page_w - 2 * margin
    avail_h = page_h - 2 * margin
    if avail_w <= 0 or avail_h <= 0:
        raise SystemExit("Margins too large for the selected page size.")

    for p in image_paths:
        # Get pixel size via Pillow (don’t convert/flatten here; ReportLab reads the file path)
        with Image.open(p) as im:
            w_px, h_px = im.size
            # Assume 72 dpi if none present; PDF coords are points (1 pt = 1/72 inch)
            dpi = im.info.get("dpi", (72, 72))[0] or 72
            img_w_in = w_px / dpi
            img_h_in = h_px / dpi
            img_w_pt = img_w_in * 72
            img_h_pt = img_h_in * 72

        # Scale to fit within available area preserving aspect
        scale = min(avail_w / img_w_pt, avail_h / img_h_pt, 1.0)
        draw_w = img_w_pt * scale
        draw_h = img_h_pt * scale
        x = (page_w - draw_w) / 2
        y = (page_h - draw_h) / 2

        c.drawImage(str(p), x, y, width=draw_w, height=draw_h, preserveAspectRatio=True, anchor='c')
        c.showPage()
    c.save()

# -------- CLI --------

def main():
    ap = argparse.ArgumentParser(description="Combine PNGs into a multi-page PDF.")
    ap.add_argument("-i", "--input-dir", type=Path, default=Path("."), help="Folder containing PNG files (default: current directory)")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output PDF path (e.g., out.pdf)")
    ap.add_argument("--pattern", default="*.png", help="Filename glob (default: *.png)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--sort", choices=["name", "mtime", "numeric"], default="name", help="Sort order (default: name)")
    ap.add_argument("--reverse", action="store_true", help="Reverse the sort order")
    ap.add_argument("--dpi", type=int, default=None, help="Embed output DPI metadata (affects printed size, Pillow mode)")
    ap.add_argument("--page-size", choices=["Letter", "A4"], help="Use ReportLab with a uniform page size")
    ap.add_argument("--margin", type=float, default=0.5, help="Page margin in inches (ReportLab mode; default 0.5)")

    args = ap.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    files = list_pngs(args.input_dir, args.pattern, args.recursive)
    if not files:
        raise SystemExit("No PNG files found.")

    files = sort_files(files, args.sort, args.reverse)

    if args.page_size:
        # ReportLab path: fit each PNG to a uniform page with margins
        save_with_reportlab(files, args.output, args.page_size, args.margin)
        print(f"[OK] Wrote {len(files)} pages to {args.output} (ReportLab, {args.page_size})")
        return

    # Pillow path: preserve each image’s native pixel size; flatten alpha to white
    images_rgb = [open_as_rgb(p) for p in files]
    save_with_pillow(images_rgb, args.output, args.dpi)
    print(f"[OK] Wrote {len(images_rgb)} pages to {args.output} (Pillow)")

if __name__ == "__main__":
    main()