"""report/report.md -> report/report.pdf via headless Chrome (no LaTeX needed)."""
import shutil
import subprocess
from pathlib import Path

import markdown

HERE = Path(__file__).parent
CSS = """
@page { size: A4; margin: 16mm 17mm; }
body { font: 10pt/1.42 "Charter", "Georgia", serif; color: #000; text-align: left; }
a { color: #000; } h1, h2, h3 { color: #000; text-align: left; }
h1 { font-size: 17pt; margin: 0 0 2pt; } h2 { font-size: 12.5pt; margin: 13pt 0 4pt; }
h3 { font-size: 10.5pt; margin: 9pt 0 3pt; }
code { font-size: 8.6pt; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 8.8pt; width: 100%; }
th, td { border-bottom: 1px solid #000; padding: 2.5pt 5pt; text-align: left; vertical-align: top; }
th { border-bottom: 1.5px solid #000; }
img { width: 100%; margin: 4pt 0; } .byline { color: #000; margin-bottom: 10pt; }
"""
CHROME = next(p for p in ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                          shutil.which("google-chrome") or "", shutil.which("chromium") or ""] if p and Path(p).exists())

body = markdown.markdown((HERE / "report.md").read_text(), extensions=["tables", "fenced_code"])
html = HERE / "report.html"
html.write_text(f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>")
subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={HERE / 'report.pdf'}", html.resolve().as_uri()], check=True, capture_output=True)
html.unlink()
print("wrote", HERE / "report.pdf")
