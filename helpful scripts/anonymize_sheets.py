#!/usr/bin/env python3
"""
anonymize_sheets.py

Crop the top portion of scanned bubble-sheet pages (to remove identifying info),
and optionally attach a fake header (generated text or a supplied image).

Dependencies:
  pip install pillow pymupdf
"""
import argparse
import os
import random
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF

def pil_from_fitz_pix(pix):
    mode = "RGB" if pix.alpha == 0 else "RGBA"
    return Image.frombytes(mode, [pix.width, pix.height], pix.samples)

def generate_fake_header(width: int, height: int, text: str,
                         font_path: Optional[str], font_size: int,
                         bg_color: str = "white", text_color: str = "black") -> Image.Image:
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    # wrap text if it's long
    margin = 10
    lines = []
    words = text.split()
    cur = ""
    for w in words:
        if draw.textsize(cur + " " + w, font=font)[0] + 2*margin > width and cur:
            lines.append(cur.strip())
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    # center vertically
    total_h = sum(draw.textsize(line, font=font)[1] for line in lines)
    y = (height - total_h) // 2
    for line in lines:
        w_line, h_line = draw.textsize(line, font=font)
        x = (width - w_line) // 2
        draw.text((x, y), line, font=font, fill=text_color)
        y += h_line
    return img

def process_pdf_pages_in_memory(doc_in: fitz.Document,
                                top_fraction: float,
                                header_template_image: Optional[Image.Image],
                                header_text_template: str,
                                random_id: bool,
                                font_path: Optional[str],
                                font_size: int,
                                bg_color: str,
                                text_color: str) -> List[Image.Image]:
    out_images = []
    page_count = doc_in.page_count
    rng = random.Random(42)
    for p in range(page_count):
        page = doc_in.load_page(p)
        pix = page.get_pixmap()  # default: RGB, 96 dpi-ish, but consistent rendering per-page
        pil = pil_from_fitz_pix(pix).convert("RGB")
        W, H = pil.size
        top_h = int(round(H * top_fraction))
        bottom_region = pil.crop((0, top_h, W, H))

        # prepare header area
        if header_template_image:
            # resize header template to width x top_h while maintaining aspect ratio (centered vertically)
            hdr = header_template_image.copy().convert("RGB")
            hdr_w, hdr_h = hdr.size
            if hdr_w != W or hdr_h != top_h:
                hdr = hdr.resize((W, top_h), resample=Image.LANCZOS)
        else:
            # generate header text (allow {page} and {anon_id})
            anon_id = f"ANON-{rng.randint(10000,99999)}" if random_id else "ANONYMIZED"
            text = header_text_template.format(page=p+1, anon_id=anon_id)
            hdr = generate_fake_header(W, top_h, text, font_path, font_size, bg_color, text_color)

        # compose into final page image
        new_page = Image.new("RGB", (W, H), color=bg_color)
        new_page.paste(hdr, (0, 0))
        new_page.paste(bottom_region, (0, top_h))
        out_images.append(new_page)
    return out_images

def save_images_as_pdf(images: List[Image.Image], out_pdf_path: str):
    if not images:
        raise ValueError("No images to save.")
    # PIL save multipage PDF (ensure RGB)
    imgs = [im.convert("RGB") for im in images]
    imgs[0].save(out_pdf_path, save_all=True, append_images=imgs[1:], resolution=300)

def main():
    ap = argparse.ArgumentParser(description="Anonymize bubble-sheet scans (crop top and add fake header).")
    ap.add_argument("--input", "-i", required=True, help="Input PDF file (multipage) or directory of images.")
    ap.add_argument("--output", "-o", required=True, help="Output anonymized PDF path.")
    ap.add_argument("--top-fraction", type=float, default=0.55,
                    help="Fraction of page height that contains identifying info and will be replaced (default 0.55).")
    ap.add_argument("--header-template", default=None,
                    help="Optional image file to use as header (resized to fit).")
    ap.add_argument("--header-text", default="ANONYMIZED — Page {page} — {anon_id}",
                    help="Template for generated header text (supports {page} and {anon_id}). Used if --header-template is not supplied.")
    ap.add_argument("--random-id", action="store_true", help="Generate a random ANON-##### id per page instead of a static ANONYMIZED label.")
    ap.add_argument("--font-path", default=None, help="Path to a TTF font to use for generated header text (optional).")
    ap.add_argument("--font-size", type=int, default=24, help="Font size for generated header (default 24).")
    ap.add_argument("--bg-color", default="white", help="Header background color (default white).")
    ap.add_argument("--text-color", default="black", help="Header text color (default black).")
    args = ap.parse_args()

    # Load header template image if provided
    header_img = None
    if args.header_template:
        header_img = Image.open(args.header_template)

    # Load input PDF or images
    images_out = []
    if os.path.isdir(args.input):
        # load images from directory in alphanumeric order
        files = sorted([os.path.join(args.input, f) for f in os.listdir(args.input)
                        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp"))])
        if not files:
            raise SystemExit("No images found in directory.")
        for fp in files:
            pil = Image.open(fp).convert("RGB")
            W, H = pil.size
            top_h = int(round(H * args.top_fraction))
            bottom_region = pil.crop((0, top_h, W, H))
            if header_img:
                hdr = header_img.resize((W, top_h), resample=Image.LANCZOS)
            else:
                anon_id = f"ANON-{random.randint(10000,99999)}" if args.random_id else "ANONYMIZED"
                text = args.header_text.format(page=os.path.basename(fp), anon_id=anon_id)
                hdr = generate_fake_header(W, top_h, text, args.font_path, args.font_size, args.bg_color, args.text_color)
            new_page = Image.new("RGB", (W, H), color=args.bg_color)
            new_page.paste(hdr, (0, 0))
            new_page.paste(bottom_region, (0, top_h))
            images_out.append(new_page)
    else:
        # assume PDF input; use PyMuPDF to render pages
        doc = fitz.open(args.input)
        images_out = process_pdf_pages_in_memory(doc, args.top_fraction, header_img,
                                                 args.header_text, args.random_id,
                                                 args.font_path, args.font_size,
                                                 args.bg_color, args.text_color)

    # Save to single multipage PDF
    save_images_as_pdf(images_out, args.output)
    print(f"Saved anonymized PDF to {args.output}. Total pages: {len(images_out)}")

if __name__ == "__main__":
    main()