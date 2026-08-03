import tempfile
import unittest
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run"))
from pair_utils import discover_pairs


class DiscoverPairsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input = self.root / "input"
        self.target = self.root / "target"
        self.input.mkdir()
        self.target.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_matches_common_stems_across_extensions(self):
        (self.input / "b.jpeg").touch()
        (self.input / "a.jpg").touch()
        (self.input / "input_only.png").touch()
        (self.target / "a.png").touch()
        (self.target / "b.webp").touch()
        (self.target / "target_only.jpg").touch()

        pairs = discover_pairs(self.input, self.target)

        self.assertEqual([pair.stem for pair in pairs], ["a", "b"])
        self.assertEqual(pairs[0].input_path.suffix, ".jpg")
        self.assertEqual(pairs[0].target_path.suffix, ".png")

    def test_rejects_duplicate_stems(self):
        (self.input / "same.jpg").touch()
        (self.input / "same.png").touch()
        (self.target / "same.png").touch()

        with self.assertRaisesRegex(ValueError, "duplicate image stems"):
            discover_pairs(self.input, self.target)


if __name__ == "__main__":
    unittest.main()
