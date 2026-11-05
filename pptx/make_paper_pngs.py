"""Usage: python -I make_paper_pngs.py <gobin.pdf> <tian.pdf> <breizh.pdf>  -> papers/gobin.png, tian.png, breizh.png (page 1, for the baseline slide)."""
import sys, os, fitz
os.makedirs("papers", exist_ok=True)
for name, path in zip(("gobin", "tian", "breizh"), sys.argv[1:4]):
    d = fitz.open(path); pm = d[0].get_pixmap(dpi=170); pm.save(f"papers/{name}.png")
    print(name, len(d), "pages;", d[0].get_text()[:90].replace("\n", " | "))
