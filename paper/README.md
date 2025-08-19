# Emergent Machine Pedagogy — Self-contained Dissertation (LaTeX)

## Build
```bash
latexmk -pdf main.tex
bibtex main
latexmk -pdf main.tex
```

## Notes
- Class: standard `report` (no external thesis class).
- Theorem/definition environments included.
- TikZ `positioning` library enabled (fixes `of` syntax).
- Replace illustrative plots/tables with your true numbers from `po.pdf` via `analyze_results.py` outputs.