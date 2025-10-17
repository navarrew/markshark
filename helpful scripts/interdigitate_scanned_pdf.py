#!/usr/bin/env python

#this script takes two scanned pdfs that were created by scanning the front and 
#back sides of the same document and merges them in the correct order.
#You should save the forward scan as A.pdf
#You should save the reverse scan as B.pdf
#It will output a new pdf called C.pdf  
#DO NOT MANUALLY REORDER THE PAGES BEFORE SCANNING THE BACK.

# to use this script you should install PyMuPDF if you haven't already
# use:  pip install pymupdf 
# (do NOT install fitz instead...that is a deprecated package)


import fitz  # fitz is the package call name for PyMuPDF

def merge_pdfs_interdigitated(pdf_a_path, pdf_b_path, output_pdf_path):
    # Open the PDF files
    pdf_a = fitz.open(pdf_a_path)
    pdf_b = fitz.open(pdf_b_path)

    # Reverse the pages of pdf_b
    pdf_b_pages = [pdf_b[i] for i in range(len(pdf_b) - 1, -1, -1)]

    # Create a new PDF for the output
    output_pdf = fitz.open()

    # Interdigitate the pages
    for page_a, page_b in zip(pdf_a, pdf_b_pages):
        output_pdf.insert_pdf(pdf_a, from_page=page_a.number, to_page=page_a.number)
        output_pdf.insert_pdf(pdf_b, from_page=page_b.number, to_page=page_b.number)

    # Save the output PDF
    output_pdf.save(output_pdf_path)

# Paths to the input and output PDF files
pdf_a_path = "A.pdf"
pdf_b_path = "B.pdf"
output_pdf_path = "C.pdf"

# Merge the PDFs
merge_pdfs_interdigitated(pdf_a_path, pdf_b_path, output_pdf_path)

print(f"The merged PDF has been saved as {output_pdf_path}.")