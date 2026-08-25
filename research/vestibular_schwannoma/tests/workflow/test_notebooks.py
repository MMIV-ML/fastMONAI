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

    def test_notebooks_use_the_project_inference_policy(self):
        sources = {
            path.name: "\n".join(self.code_cells(path)) for path in NOTEBOOKS
        }
        training_source = sources["01_five_fold_cross_validation.ipynb"]
        self.assertIn("use_tta=True", training_source)
        self.assertIn("training_seed=42", training_source)
        self.assertNotIn("all_data_monitor_seed", training_source)
        self.assertIn("USE_TTA = True", sources["02_inference_new_cases.ipynb"])
        for source in sources.values():
            self.assertNotIn("use_tta=False", source)
            self.assertNotIn("USE_TTA = False", source)


if __name__ == "__main__":
    unittest.main()
