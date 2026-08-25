# Repository Guidelines

## nbdev workflow

- Treat `nbs/*.ipynb` as the source of truth for nbdev-exported package code.
- Do not edit generated `fastMONAI/*.py` modules directly.
- After changing library notebooks, run `nbdev_prepare` from the repository root to regenerate modules and run checks.
