import subprocess
import sys
from pathlib import Path

def compress_pdf(infile, outfile=None, quality="ebook"):
    """
    Compress PDF using Ghostscript.

    quality options:
      - screen  (smallest, 72 dpi)
      - ebook   (good balance, ~150 dpi)
      - printer (higher dpi, larger file)
      - prepress (high quality, large)
    """
    infile = Path(infile)
    if outfile is None:
        outfile = infile.with_name(infile.stem + "_small.pdf")

    cmd = [
        "gs",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        f"-dPDFSETTINGS=/{quality}",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        f"-sOutputFile={outfile}",
        str(infile),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Compressed file written to:", outfile)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compress_pdf.py input.pdf [quality]")
        sys.exit(1)

    infile = sys.argv[1]
    quality = sys.argv[2] if len(sys.argv) > 2 else "ebook"
    compress_pdf(infile, quality=quality)