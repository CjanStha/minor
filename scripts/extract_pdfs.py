import sys
from pathlib import Path

try:
    from PyPDF2 import PdfReader
except Exception as e:
    print('PyPDF2 not installed', e)
    raise

BASE = Path('notesForMP')
files = {
    'proposal': BASE / 'mpProposal.pdf',
    'midterm': BASE / 'Mid-term_Minor_project_report.pdf'
}

for key, path in files.items():
    out = Path('notesForMP') / f'{key}_text.txt'
    if not path.exists():
        print(f'Missing: {path}')
        continue
    reader = PdfReader(str(path))
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            texts.append(page.extract_text() or '')
        except Exception as e:
            texts.append('')
    out.write_text('\n\n'.join(texts), encoding='utf-8')
    print(f'Wrote {out}')
