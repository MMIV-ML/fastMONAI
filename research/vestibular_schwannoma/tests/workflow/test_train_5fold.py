import unittest

from vestibular_schwannoma import train_5fold


class FiveFoldLauncherTests(unittest.TestCase):
    def test_defaults_select_all_models_and_preserve_80_20_control(self):
        args = train_5fold._parser().parse_args([])
        self.assertEqual(train_5fold.DEFAULT_MODELS, ("unet", "dynunet", "segmamba"))

        self.assertEqual(tuple(args.models), train_5fold.DEFAULT_MODELS)
        self.assertEqual(args.folds, [1, 2, 3, 4, 5])
        self.assertEqual(args.foreground_probability, 0.8)
        self.assertNotIn("use_tta", vars(args))

    def test_70_30_sampling_remains_an_explicit_comparison(self):
        args = train_5fold._parser().parse_args(["--foreground-probability", "0.7"])

        self.assertEqual(args.foreground_probability, 0.7)


if __name__ == "__main__":
    unittest.main()
