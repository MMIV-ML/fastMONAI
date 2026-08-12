import ast
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = (
    PROJECT_ROOT / "notebooks" / "01_five_fold_cross_validation.ipynb",
    PROJECT_ROOT / "notebooks" / "02_inference_new_cases.ipynb",
)


class NotebookStructureTests(unittest.TestCase):
    def code_cells(self, path):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        return [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]

    def test_every_code_cell_is_valid_python(self):
        for path in NOTEBOOKS:
            for index, source in enumerate(self.code_cells(path)):
                with self.subTest(notebook=path.name, cell=index):
                    compile(source, f"{path.name}:cell-{index}", "exec")

    def test_notebooks_do_not_reintroduce_local_helpers(self):
        for path in NOTEBOOKS:
            definitions = []
            for source in self.code_cells(path):
                tree = ast.parse(source)
                definitions.extend(
                    node.name
                    for node in ast.walk(tree)
                    if isinstance(
                        node,
                        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                    )
                )
            self.assertEqual(definitions, [], path.name)


if __name__ == "__main__":
    unittest.main()
