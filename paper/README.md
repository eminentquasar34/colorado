# Paper Workflow

The report source lives in `paper/main.tex`.

## View The Structure

You can inspect the paper outline without opening Overleaf or compiling:

```bash
python3 scripts/latex_outline.py paper/main.tex
```

This prints sections, subsections, labels, captions, and rough word counts.

## Compile The Paper

This file currently uses the CVPR author kit:

```tex
\usepackage[review]{cvpr}
```

To compile locally, install a LaTeX distribution and make sure the CVPR style
files are available. Options:

- Use Overleaf with the CVPR template files.
- Install MacTeX locally and copy `cvpr.sty` / bibliography style files into
  `paper/`.
- Install `tectonic` or TinyTeX if you prefer a smaller setup, then add the CVPR
  style files.

Once a compiler is installed and the style files are present:

```bash
cd paper
latexmk -pdf main.tex
```

Generated PDFs and auxiliary LaTeX files should not be committed.
