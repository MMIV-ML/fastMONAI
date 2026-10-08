# Repository Guidelines

## nbdev workflow

- Treat `nbs/*.ipynb` as the source of truth for nbdev-exported package code.
- Do not edit generated `fastMONAI/*.py` modules directly.
- After changing library notebooks, run `nbdev_prepare` from the repository root to regenerate modules and run checks.

## Commit messages

- Use Conventional Commits: `<type>(<optional scope>): <summary>`, for example
  `fix(vs-pacs): publish pr2mask mask series`.
- Types:
  - `feat`: new functionality
  - `fix`: bug fix
  - `docs`: documentation only
  - `test`: tests only
  - `refactor`: code change without a behaviour change
  - `perf`: performance improvement
  - `build`: dependencies, packaging or Docker images
  - `ci`: CI configuration
  - `chore`: maintenance that fits none of the above, such as `.gitignore`
- There is no `bug` or `minor` type: use `fix` for bugs, and `chore`, `docs` or
  `refactor` for small changes.
- Mark breaking changes with `!` after the type (`feat!: ...`) and explain them in the body.
- Write the summary in the imperative, lower case, without a trailing period and
  under about 72 characters. Use the body to explain why.
- Pull requests to `main` are squash-merged, so give the pull request title the same format.
