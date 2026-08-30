import tempfile
import threading
import unittest
from pathlib import Path

import pandas as pd

from scieval.dataset.SciCode.scicode import SciCode
from scieval.inference import infer_data_api


def _sub_step(number):
    return {
        "step_number": number,
        "step_description_prompt": f"description-{number}",
        "step_background": f"background-{number}",
        "function_header": f"def function_{number}():",
        "return_line": "return None",
    }


class SciCodePromptContextTest(unittest.TestCase):
    def _dataset(self, problem_id, step_count):
        dataset = object.__new__(SciCode)
        dataset.split = "test"
        dataset.with_background = True
        dataset.prompt_template = (
            "PREVIOUS\n{problem_steps_str}\n"
            "NEXT\n{next_step_str}\n"
            "DEPENDENCIES\n{dependencies}"
        )
        record = {
            "problem_id": str(problem_id),
            "sub_steps": [_sub_step(i) for i in range(1, step_count + 1)],
            "required_dependencies": "import numpy as np",
        }
        return dataset, record

    def test_later_prompt_contains_prior_generated_code_in_order(self):
        dataset, record = self._dataset("1", 3)
        row = pd.Series({"record": record, "step": 3})

        prompt = dataset.build_prompt_with_context(
            row,
            {
                1: "```python\nimport numpy as np\ndef function_1():\n    return 1\n```",
                2: "```python\ndef function_2():\n    return function_1() + 1\n```",
            },
        )[0]["value"]

        expected = (
            "description-1\nbackground-1\n\n"
            "def function_1():\n    return 1\n\n"
            "------\n\n"
            "description-2\nbackground-2\n\n"
            "def function_2():\n    return function_1() + 1"
        )
        self.assertIn(expected, prompt)
        self.assertNotIn("import numpy as np\ndef function_1", prompt)
        self.assertLess(prompt.index("function_1"), prompt.index("function_2"))

    def test_excluded_dependency_step_uses_curated_implementation(self):
        dataset, record = self._dataset("13", 7)
        row = pd.Series({"record": record, "step": 7})
        previous = {
            step: f"```python\ndef function_{step}():\n    return {step}\n```"
            for step in range(1, 6)
        }

        prompt = dataset.build_prompt_with_context(row, previous)[0]["value"]

        self.assertIn("class Maxwell", prompt)
        self.assertIn("description-6\nbackground-6", prompt)


class _SequentialDataset:
    dataset_name = "SciCode"
    SEQUENTIAL_INFERENCE = True

    def __init__(self):
        self.data = pd.DataFrame(
            [
                {"index": 0, "id": "1.1", "problem_id": "1", "step": 1},
                {"index": 1, "id": "1.2", "problem_id": "1", "step": 2},
                {"index": 2, "id": "2.1", "problem_id": "2", "step": 1},
                {"index": 3, "id": "2.2", "problem_id": "2", "step": 2},
            ]
        )

    def build_prompt_with_context(self, row, previous):
        step = int(row["step"])
        if step == 1:
            assert previous == {}
        else:
            assert previous == {1: f"response-{row['problem_id']}.1"}
        value = f"{row['id']}|previous={previous}"
        return [{"type": "text", "value": value}]


class _RecordingModel:
    is_api = True

    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def generate(self, message, dataset):
        step_id = message[0]["value"].split("|", 1)[0]
        with self.lock:
            self.calls.append(step_id)
        return f"response-{step_id}"


class SciCodeSequentialInferenceTest(unittest.TestCase):
    def test_steps_are_ordered_with_cross_problem_parallelism(self):
        dataset = _SequentialDataset()
        model = _RecordingModel()

        with tempfile.TemporaryDirectory() as temp_dir:
            results = infer_data_api(
                model=model,
                work_dir=temp_dir,
                model_name="test-model",
                dataset=dataset,
                index_set={0, 1, 2, 3},
                api_nproc=2,
                ignore_failed=False,
                existing_results={},
            )

            self.assertFalse(
                Path(temp_dir, "test-model_SciCode_sequential_supp.pkl").exists()
            )

        self.assertEqual(
            results,
            {
                0: "response-1.1",
                1: "response-1.2",
                2: "response-2.1",
                3: "response-2.2",
            },
        )
        first_second_step = min(model.calls.index("1.2"), model.calls.index("2.2"))
        self.assertLess(model.calls.index("1.1"), first_second_step)
        self.assertLess(model.calls.index("2.1"), first_second_step)


if __name__ == "__main__":
    unittest.main()
