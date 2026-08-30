import unittest
import importlib.util
from pathlib import Path


_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "scieval"
    / "dataset"
    / "SciCode"
    / "protocol.py"
)
_SPEC = importlib.util.spec_from_file_location("scicode_protocol", _PROTOCOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
protocol = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(protocol)


class SciCodeProtocolTest(unittest.TestCase):
    def test_official_step_counts_and_exclusions(self):
        self.assertEqual(protocol.OFFICIAL_TEST_STEP_COUNT, 288)
        self.assertEqual(protocol.OFFICIAL_VALIDATION_STEP_COUNT, 50)
        self.assertEqual(
            protocol.OFFICIAL_EXCLUDED_TEST_STEPS,
            {"13.6", "62.1", "76.3"},
        )
        self.assertFalse(protocol.is_official_scored_step("test", "13", 6))
        self.assertFalse(protocol.is_official_scored_step("test", "62", 1))
        self.assertFalse(protocol.is_official_scored_step("test", "76", 3))
        self.assertTrue(protocol.is_official_scored_step("validation", "13", 6))
        self.assertTrue(protocol.is_official_scored_step("test", "13", 5))

    def test_curated_dependencies_are_packaged(self):
        self.assertIn("class Maxwell", protocol.official_dependency_code("13.6"))
        self.assertIn("class EnlargedBlock", protocol.official_dependency_code("62.1"))
        self.assertIn("def generate_dna", protocol.official_dependency_code("76.3"))
        with self.assertRaises(KeyError):
            protocol.official_dependency_code("13.5")

    def test_extract_python_script_matches_multistep_context_contract(self):
        response = """Explanation
```python
import numpy as np
from scipy import optimize

def solve():
    return 42
```
"""
        self.assertEqual(
            protocol.extract_python_script(response),
            "def solve():\n    return 42",
        )
        self.assertEqual(protocol.extract_python_script(None), "")
        self.assertEqual(protocol.extract_python_script("nan"), "")


if __name__ == "__main__":
    unittest.main()
