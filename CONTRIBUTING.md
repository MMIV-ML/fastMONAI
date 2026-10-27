# How to contribute
Contributions to the source code are greatly appreciated. If you find any issues or have suggestions for improvements, please open an issue in the corresponding GitHub repository. To contribute code changes, we encourage you to follow these [steps](https://docs.github.com/en/get-started/quickstart/contributing-to-projects). 

## Getting Started
1. Install an editable version of the forked project from inside the project folder: `pip install -e '.[dev]'`
2. Git hooks are used to run automatic scripts during each commit and merge to strip the notebooks of superfluous metadata and avoid merge conflicts. To set up Git hooks, run the following command inside the project folder: `nbdev_install_hooks`
3. Once you have installed the project and set up the Git hooks, you can start making changes to the notebooks.
4. Run: `nbdev_prepare` to build .py modules from notebooks and test them (if unit tests are written). 
5. Write commit messages and pull request titles as [Conventional Commits](https://www.conventionalcommits.org/), for example `fix(vision_patch): handle empty masks`. See `AGENTS.md` for the allowed types.

## Contact
If you have any questions or need further assistance, please contact us at [skaliyugarasan@hotmail.com]. We appreciate your interest and look forward to your contributions.