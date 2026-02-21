import re
import json
from pathlib import Path

SRC = Path('notesForMP/midterm_update_notebook.xml')
OUT = Path('notesForMP/midterm_update_notebook.ipynb')

text = SRC.read_text(encoding='utf-8')
pattern = re.compile(r'<VSCode.Cell(?:\s+id="(?P<id>[^"]+)")?\s+language="(?P<lang>[^"]+)">(?P<body>.*?)</VSCode.Cell>', re.S)

matches = list(pattern.finditer(text))
if not matches:
    print('No VSCode.Cell entries found in', SRC)
    raise SystemExit(1)

cells = []
for m in matches:
    lang = m.group('lang').strip()
    body = m.group('body')
    # Trim leading/trailing newlines added by XML file
    if body.startswith('\n'):
        body = body[1:]
    if body.endswith('\n'):
        body = body[:-1]
    # Normalize line endings
    lines = body.split('\n')
    # Keep as list of lines with trailing newlines for notebook format
    src_lines = [l + '\n' for l in lines]

    if lang.lower() == 'python':
        cell = {
            'cell_type': 'code',
            'metadata': {},
            'source': src_lines,
            'outputs': [],
            'execution_count': None
        }
    else:
        # treat everything else as markdown
        cell = {
            'cell_type': 'markdown',
            'metadata': {},
            'source': src_lines
        }
    cells.append(cell)

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

OUT.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print('Wrote', OUT)
