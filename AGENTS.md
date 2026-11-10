# Repository Guidelines

fastMONAI is an nbdev library: `nbs/` is exported to `fastMONAI/`. `research/` holds
application projects written as plain Python and notebooks; nbdev does not export them.

## Library (nbdev)

- `nbs/*.ipynb` is the source of truth. Don't edit the generated `fastMONAI/*.py` modules.
- After changing library notebooks, run `nbdev_prepare` from the repository root.
- Clean only the notebooks you edited (`nbdev_clean --fname <notebook>`). A repo-wide clean
  rewrites the tutorials' stored outputs.
- `README.md` is generated from `nbs/index.ipynb`; edit the notebook, not the README.

## Research projects

- Each project has its own README; see `research/README.md`.
- CI runs `python -m pytest -q research/vestibular_schwannoma/tests` from the repository root.
  Without pytest, run the deployment tests from `research/`:
  `python -m unittest discover -s vestibular_schwannoma/tests/deployment -t .`
- Container release folders hold only the image archive, a short README and
  `ror_select_entry.txt`. Don't write `.sha256` files.

## Writing

- Keep READMEs, comments and commit messages short and plain. Don't use em dashes.
- Comments say what the code does. Which bug a change fixes belongs in the commit message.

## Commit messages

- Use Conventional Commits: `<type>(<optional scope>): <summary>`, for example
  `fix(vs-pacs): publish pr2mask mask series`.
- Types: `feat`, `fix`, `docs`, `test`, `refactor` (no behaviour change), `perf`, `build`
  (dependencies, packaging, Docker images), `ci` and `chore` (anything else, such as
  `.gitignore`). There is no `bug` or `minor` type.
- Mark breaking changes with `!` after the type (`feat!: ...`) and explain them in the body.
- Write the summary in the imperative, lower case, without a trailing period and under about
  72 characters. Use the body to explain why.
- Pull requests to `main` are squash-merged, so give the pull request title the same format.
