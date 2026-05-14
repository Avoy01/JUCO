# JUCO Mini Project

This repository contains the manuscript, figures, and Python analysis code for
JUCO, a joint uncertainty-calibrated optimisation framework for ICU mortality
prediction under informative missingness.

## Contents

- `juco_main.tex` - main LaTeX manuscript.
- `juco_core.py` - shared modelling and evaluation utilities.
- `JUCO_Part1_MIMIC.py` - MIMIC-IV training/evaluation workflow.
- `JUCO_Part1B_Robustness.py` - robustness experiments.
- `JUCO_Part2_eICU.py` - eICU external validation workflow.
- `JUCO_Part3_Aggregate.py` - aggregation and reporting workflow.
- `generate_figures.py` - figure generation script.
- `Figure*.pdf`, `fig.pdf`, and `*.drawio` - manuscript figures and diagrams.

## Data

The raw MIMIC-IV and eICU files are intentionally not tracked in Git. Place
local copies under `mimic-iv/` and `eicu/` when reproducing experiments.

Generated checkpoints, logs, and large result artifacts are also ignored by
default. Keep only small, publication-ready result summaries in version control.

## Setup

Create a virtual environment and install the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Compile the manuscript with:

```powershell
pdflatex juco_main.tex
bibtex juco_main
pdflatex juco_main.tex
pdflatex juco_main.tex
```

## GitHub Push

After reviewing the tracked files, connect a remote and push:

```powershell
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```
