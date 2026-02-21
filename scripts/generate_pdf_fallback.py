from fpdf import FPDF
import json
from pathlib import Path

IN = Path('notesForMP/midterm_update_notebook.ipynb')
OUT = Path('notesForMP/midterm_update_notebook_fallback.pdf')

nb = json.loads(IN.read_text(encoding='utf-8'))

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font('Arial', 'B', 16)

# helper to coerce text into latin-1 safe form for PyFPDF
def safe_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    # replace common Unicode punctuation with ASCII equivalents
    repl = {
        '—': '-',  # em dash
        '–': '-',  # en dash
        '‘': "'", # left single quote
        '’': "'", # right single quote
        '“': '"',  # left double quote
        '”': '"',  # right double quote
        '‑': '-',
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    # finally, encode->decode using latin-1 with replacement to avoid errors
    b = s.encode('latin-1', errors='replace')
    return b.decode('latin-1')

pdf.cell(0, 8, safe_text('Mid-term Progress Report — Notebook'), ln=True)
pdf.ln(4)

for cell in nb.get('cells', []):
    ctype = cell.get('cell_type')
    src = ''.join(cell.get('source', []))
    # sanitize text for PyFPDF
    src = safe_text(src)
    if ctype == 'markdown':
        # simple markdown -> text
        pdf.set_font('Arial', '', 11)
        for line in src.split('\n'):
            line = line.strip()
            if not line:
                pdf.ln(3)
                continue
            # Bold headings
            if line.startswith('#'):
                heading = line.lstrip('#').strip()
                pdf.set_font('Arial', 'B', 13)
                pdf.multi_cell(0, 6, heading)
                pdf.ln(1)
                pdf.set_font('Arial', '', 11)
            else:
                pdf.multi_cell(0, 6, safe_text(line))
        pdf.ln(4)
    elif ctype == 'code':
        pdf.set_font('Courier', '', 9)
        pdf.set_text_color(40, 40, 40)
        # sanitize code text similarly
        code_text = safe_text(src)
        pdf.multi_cell(0, 5, code_text)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

pdf.output(str(OUT))
print('Wrote', OUT)
