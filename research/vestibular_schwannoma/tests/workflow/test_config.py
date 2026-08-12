import unittest
from dataclasses import FrozenInstanceError

from fastMONAI.vision_all import ZNormalization

from vestibular_schwannoma.workflow.config import (
    ExperimentConfig,
    make_patch_config,
)


class ExperimentConfigTests(unittest.TestCase):
    def test_defaults_preserve_the_notebook_experiment(self):
        config = ExperimentConfig()
        self.assertEqual(config.model_keys, ("unet", "dynunet", "segmamba"))
        self.assertEqual(config.folds, (1, 2, 3, 4, 5))
        self.assertEqual(config.target_spacing, (0.4102, 0.4102, 1.5))
        self.assertEqual(config.patch_size, (192, 192, 48))
        self.assertEqual(config.epochs, 500)
        self.assertEqual(config.batch_size, 4)

    def test_configuration_is_frozen(self):
        config = ExperimentConfig(model_keys=("unet",))
        with self.assertRaises(FrozenInstanceError):
            config.epochs = 2

    def test_invalid_declarations_fail_early(self):
        cases = [
            {"model_keys": ()},
            {"model_keys": ("unet", "unet")},
            {"model_keys": ("unknown",)},
            {"model_keys": ("unet",), "folds": (1, 1)},
            {
                "model_keys": ("unet",),
                "run_cross_validation": False,
                "train_all_data": False,
            },
            {"model_keys": ("unet",), "epochs": 0},
            {"model_keys": ("unet",), "patch_size": (192, 192, 0)},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ExperimentConfig(**kwargs)

    def test_patch_factory_preserves_training_and_inference_contract(self):
        config = ExperimentConfig(model_keys=("unet",))
        normalization = [ZNormalization(masking_method="foreground")]
        patch = make_patch_config(config, normalization)
        self.assertEqual(patch.patch_size, [192, 192, 48])
        self.assertEqual(patch.target_spacing, [0.4102, 0.4102, 1.5])
        self.assertEqual(patch.label_probabilities, {0: 0.2, 1: 0.8})
        self.assertTrue(patch.preprocessed)
        self.assertTrue(patch.keep_largest_component)
        self.assertEqual(patch.patch_overlap, 0.5)


if __name__ == "__main__":
    unittest.main()
