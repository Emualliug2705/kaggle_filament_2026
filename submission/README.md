# Final submission

| File | What |
|---|---|
| `submission.csv` | The submitted file: Kaggle submission ref **58424919**, public leaderboard **0.42**, out-of-fold PQ **0.4801** (707 training photos). Same file as [`reference/submission.csv`](../reference/submission.csv). |
| `main.pdf` | The competition report describing this submission, a copy of [`report/main.pdf`](../report/main.pdf). |

**Report sources.** `main.pdf` is compiled from the LaTeX sources in [`report/`](../report/): [`main.tex`](../report/main.tex)
(text), [`preamble.tex`](../report/preamble.tex), [`main.bib`](../report/main.bib) (references) and
[`figures/`](../report/figures/) (Fig. 1 is drawn in TikZ by [`figures/pipeline.tex`](../report/figures/pipeline.tex)). It is built with
XeLaTeX and BibTeX, from `report/`: `xelatex main && bibtex main && xelatex main && xelatex main`. After editing the report, copy the
new `report/main.pdf` here.

**Pipeline.** [`PIPELINE.ipynb`](../PIPELINE.ipynb) rebuilds `submission.csv` from the raw photos (see the repository
[README](../README.md#reproducing-the-result), *Reproducing the result*).

SHA-256 of `submission.csv`: `d503ddac294d7cd5fd65dbe082a6c523f9178c093e1a54bc843ce39406c6a219`
