import unittest
from unittest.mock import patch

from vestibular_schwannoma.workflow import models


class ModelRecipeTests(unittest.TestCase):
    def test_specs_are_the_single_architecture_declaration(self):
        self.assertEqual(models.UNET_SPEC["arch_id"], "monai.unet")
        self.assertEqual(models.DYNUNET_SPEC["arch_id"], "monai.dynunet")
        self.assertEqual(
            models.DYNUNET_SPEC["wrapper_spec"][0]["wrapper_id"],
            "fastmonai.dynunet_ds_adapter",
        )
        self.assertEqual(models.SEGMAMBA_SPEC["arch_id"], "segmamba.v2")

    def test_declared_order_is_preserved(self):
        recipes = models.get_model_recipes(("dynunet", "unet"))
        self.assertEqual(list(recipes), ["dynunet", "unet"])

    def test_missing_optional_segmamba_can_skip_or_fail(self):
        with patch.object(models, "segmamba_available", return_value=False):
            with self.assertWarns(UserWarning):
                recipes = models.get_model_recipes(("unet", "segmamba"))
            self.assertEqual(list(recipes), ["unet"])
            with self.assertRaisesRegex(ImportError, "SegMamba was requested"):
                models.get_model_recipes(("segmamba",), skip_unavailable=False)

    def test_compilation_is_applied_after_spec_construction(self):
        sentinel_model = object()
        compiled_model = object()
        recipe = models.MODEL_RECIPES["unet"]
        with (
            patch.object(
                models, "build_model_from_spec", return_value=sentinel_model
            ) as build,
            patch.object(models.torch, "compile", return_value=compiled_model) as compile,
        ):
            result = models.build_training_model(recipe, compile_model=True)
        build.assert_called_once_with(recipe.model_spec)
        compile.assert_called_once_with(sentinel_model)
        self.assertIs(result, compiled_model)

    def test_segmamba_preserves_the_uncompiled_training_path(self):
        sentinel_model = object()
        recipe = models.MODEL_RECIPES["segmamba"]
        with (
            patch.object(
                models, "build_model_from_spec", return_value=sentinel_model
            ),
            patch.object(models.torch, "compile") as compile,
        ):
            result = models.build_training_model(recipe, compile_model=True)
        compile.assert_not_called()
        self.assertIs(result, sentinel_model)

    def test_each_build_constructs_a_fresh_model_from_the_spec(self):
        first, second = object(), object()
        recipe = models.MODEL_RECIPES["unet"]
        with patch.object(
            models, "build_model_from_spec", side_effect=[first, second]
        ) as build:
            results = [
                models.build_training_model(recipe, compile_model=False)
                for _ in range(2)
            ]
        self.assertEqual(results, [first, second])
        self.assertEqual(build.call_count, 2)


if __name__ == "__main__":
    unittest.main()
