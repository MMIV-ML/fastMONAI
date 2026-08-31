import unittest
from unittest.mock import patch

from vestibular_schwannoma.workflow import models


class TrainingModelConfigTests(unittest.TestCase):
    def test_specs_are_the_single_architecture_declaration(self):
        self.assertEqual(models.UNET_SPEC["arch_id"], "monai.unet")
        self.assertEqual(models.DYNUNET_SPEC["arch_id"], "monai.dynunet")
        self.assertEqual(
            models.DYNUNET_SPEC["wrapper_spec"][0]["wrapper_id"],
            "fastmonai.dynunet_ds_adapter",
        )
        self.assertEqual(
            models.UNET_SPEC["arch_kwargs"]["channels"],
            [32, 64, 128, 256, 320],
        )
        self.assertEqual(
            models.DYNUNET_SPEC["arch_kwargs"]["filters"],
            [32, 64, 128, 256, 320],
        )
        self.assertEqual(models.SEGMAMBA_SPEC["arch_id"], "segmamba.v2")

    def test_registry_contains_only_supported_architectures(self):
        self.assertEqual(
            set(models.TRAINING_MODEL_CONFIGS),
            {"unet", "dynunet", "segmamba"},
        )

    def test_loss_specs_declare_scientifically_relevant_parameters(self):
        self.assertEqual(
            models.TRAINING_MODEL_CONFIGS["unet"].loss_spec,
            {
                "loss_id": "monai.dice_ce",
                "kwargs": {
                    "to_onehot_y": True,
                    "softmax": True,
                    "include_background": False,
                    "batch": True,
                },
            },
        )
        self.assertEqual(
            models.TRAINING_MODEL_CONFIGS["dynunet"].loss_spec,
            {
                "loss_id": "monai.deep_supervision",
                "kwargs": {"weight_mode": "exp"},
                "base_loss": models.DICE_CE_LOSS_SPEC,
            },
        )

    def test_declared_order_is_preserved(self):
        configs = models.get_training_model_configs(("dynunet", "unet"))
        self.assertEqual(list(configs), ["dynunet", "unet"])

    def test_registry_keys_and_conventional_names_are_consistent(self):
        for key, config in models.TRAINING_MODEL_CONFIGS.items():
            with self.subTest(key=key):
                self.assertEqual(config.key, key)
                self.assertEqual(config.checkpoint_name, f"best_{key}")
                self.assertEqual(
                    config.experiment_name,
                    f"vestibular_schwannoma_{key}",
                )

    def test_missing_optional_segmamba_can_skip_or_fail(self):
        with patch.object(models, "segmamba_available", return_value=False):
            with self.assertWarns(UserWarning):
                configs = models.get_training_model_configs(("unet", "segmamba"))
            self.assertEqual(list(configs), ["unet"])
            with self.assertRaisesRegex(ImportError, "SegMamba was requested"):
                models.get_training_model_configs(("segmamba",), skip_unavailable=False)

    def test_compilation_is_applied_after_spec_construction(self):
        sentinel_model = object()
        compiled_model = object()
        config = models.TRAINING_MODEL_CONFIGS["unet"]
        with (
            patch.object(
                models, "build_model_from_spec", return_value=sentinel_model
            ) as build,
            patch.object(
                models.torch, "compile", return_value=compiled_model
            ) as compile,
        ):
            result = models.build_training_model(config, compile_model=True)
        build.assert_called_once_with(config.model_spec)
        compile.assert_called_once_with(sentinel_model)
        self.assertIs(result, compiled_model)

    def test_segmamba_preserves_the_uncompiled_training_path(self):
        sentinel_model = object()
        config = models.TRAINING_MODEL_CONFIGS["segmamba"]
        with (
            patch.object(models, "build_model_from_spec", return_value=sentinel_model),
            patch.object(models.torch, "compile") as compile,
        ):
            result = models.build_training_model(config, compile_model=True)
        compile.assert_not_called()
        self.assertIs(result, sentinel_model)

    def test_each_build_constructs_a_fresh_model_from_the_spec(self):
        first, second = object(), object()
        config = models.TRAINING_MODEL_CONFIGS["unet"]
        with patch.object(
            models, "build_model_from_spec", side_effect=[first, second]
        ) as build:
            results = [
                models.build_training_model(config, compile_model=False)
                for _ in range(2)
            ]
        self.assertEqual(results, [first, second])
        self.assertEqual(build.call_count, 2)


if __name__ == "__main__":
    unittest.main()
