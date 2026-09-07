"""Package only the manuscript's reachable sources and figures for arXiv."""
from pathlib import Path
import hashlib
import json
import re
import tarfile

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'paper-arxiv'


def main():
    selected = set()

    def add(relative):
        path = (PAPER / relative).resolve()
        if not path.is_relative_to(PAPER) or not path.is_file():
            raise ValueError(f'Missing or external dependency: {relative}')
        if path in selected:
            return
        selected.add(path)
        if path.suffix not in ('.tex', '.sty'):
            return
        source = re.sub(r'(?<!\\)%[^\n]*', '', path.read_text())
        for command, argument in re.findall(
                r'\\(input|include|includegraphics|bibliography|usepackage|bibliographystyle)'
                r'(?:\[[^\]]*\])?\{([^}]+)\}', source):
            for name in argument.split(','):
                suffix = {'input': '.tex', 'include': '.tex', 'bibliography': '.bib',
                          'usepackage': '.sty', 'bibliographystyle': '.bst'}.get(command)
                dependency = Path(name.strip())
                if suffix and not dependency.suffix:
                    dependency = dependency.with_suffix(suffix)
                if command in ('usepackage', 'bibliographystyle') and not (PAPER / dependency).exists():
                    continue  # Standard TeX distribution dependency.
                add(dependency)

    add('main.tex')
    add('main.bbl')
    pdf = PAPER / 'main.pdf'
    if not pdf.is_file() or any(p.stat().st_mtime > pdf.stat().st_mtime for p in selected):
        raise RuntimeError('Rebuild paper-arxiv/main.pdf before packaging the edited sources.')
    # Keep the text-box abstract synchronized with the exact manuscript source.
    abstract = (PAPER / 'sections/abstract.tex').read_text()
    abstract = abstract.replace('{,}', ',').replace(r'\$', '$').replace(r'\%', '%').replace(r'\&', '&')
    abstract = abstract.replace('---', '—').replace('--', '–')
    if re.search(r'\\[A-Za-z]+|[{}]', abstract):
        raise ValueError('The abstract contains unsupported LaTeX; extend the plain-text conversion explicitly.')
    abstract = ' '.join(abstract.split()) + '\n'
    (PAPER / 'abstract-v2.1.txt').write_text(abstract)
    archive = PAPER / 'arxiv-source-v2.1.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for path in sorted(selected):
            bundle.add(path, arcname=str(path.relative_to(PAPER)), recursive=False)
    manifest = {
        'archive': str(archive.relative_to(ROOT)),
        'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'reviewed_pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
        'plain_abstract_sha256': hashlib.sha256(abstract.encode()).hexdigest(),
        'files': {str(p.relative_to(PAPER)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(selected)},
    }
    destination = ROOT / 'code/results/chronological_validation/arxiv_source_manifest.json'
    destination.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{archive}: {len(selected)} files, {archive.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
