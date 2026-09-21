#!/usr/bin/env python3
"""Build the current public v3 results page from its aggregate data export."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def build():
    data = json.loads((ROOT / 'site/data/transfer_results.json').read_text())
    assert len(data['models']) == 4
    assert sum(len(m['rows']) for m in data['models'].values()) == 24
    template = (ROOT / 'scripts/results_page_template.html').read_text()
    assert template.count('/*__DATA__*/') == 1
    page = template.replace('/*__DATA__*/', json.dumps(data, ensure_ascii=False).replace('</', '<\\/'))
    out = ROOT / 'site/index.html'
    temp = out.with_name('index.html.pending')
    temp.write_text(page)
    temp.replace(out)
    print(f'Built {out}: {len(page)} characters, 4 models, 24 evaluations')

if __name__ == '__main__':
    build()
