import fitz
import os
import glob

base_dir = '/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Paper/latex_source/figures'
out_dir = '/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Paper/images'

pdfs = glob.glob(os.path.join(base_dir, '*.pdf'))
for pdf in pdfs:
    doc = fitz.open(pdf)
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    name = os.path.basename(pdf).replace('.pdf', '.png')
    out_path = os.path.join(out_dir, name)
    pix.save(out_path)
    print(f"Saved {out_path}")
