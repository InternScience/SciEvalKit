"""Constants and helpers for the official SciCode evaluation protocol."""

from pathlib import Path
import re


# These raw Hugging Face test records are dependency-only steps.  The official
# generator and evaluator skip them, yielding 288 scored test subproblems from
# the 291 raw ``sub_steps``.  Later steps consume the curated implementations
# stored under ``official_dependencies`` instead of model generations.
OFFICIAL_EXCLUDED_TEST_STEPS = frozenset({"13.6", "62.1", "76.3"})

OFFICIAL_TEST_PROBLEM_COUNT = 65
OFFICIAL_VALIDATION_PROBLEM_COUNT = 15
OFFICIAL_TEST_STEP_COUNT = 288
OFFICIAL_VALIDATION_STEP_COUNT = 50

_DEPENDENCY_DIR = Path(__file__).resolve().parent / "official_dependencies"


def make_step_id(problem_id: object, one_based_step: int) -> str:
    """Return the canonical ``{problem}.{step}`` identifier."""

    return f"{problem_id}.{one_based_step}"


def is_official_scored_step(
    split: str, problem_id: object, one_based_step: int
) -> bool:
    """Whether a raw SciCode sub-step is scored by the official protocol."""

    return not (
        split == "test"
        and make_step_id(problem_id, one_based_step)
        in OFFICIAL_EXCLUDED_TEST_STEPS
    )


def official_dependency_code(step_id: str) -> str:
    """Load the official curated implementation for a dependency-only step."""

    if step_id not in OFFICIAL_EXCLUDED_TEST_STEPS:
        raise KeyError(f"SciCode step {step_id} is not an excluded dependency step")
    path = _DEPENDENCY_DIR / f"{step_id}.py"
    return path.read_text(encoding="utf-8")


def extract_python_script(response: object) -> str:
    """Extract code using SciCode's response convention.

    The upstream implementation removes imports because dependencies are
    supplied separately when the generated programs are assembled.  Empty and
    failed API responses are normalized to an empty function body so they do
    not leak error messages (or pandas' ``nan`` sentinel) into later prompts.
    """

    if response is None:
        return ""
    text = str(response).strip()
    if not text or text.lower() == "nan" or "Failed to obtain answer via API." in text:
        return ""

    if "```" in text:
        if "```python" in text:
            text = text.split("```python", 1)[1].split("```", 1)[0]
        else:
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text.replace("```", "")

    return re.sub(
        r"^\s*(import .*|from .*\s+import\s+.*)",
        "",
        text,
        flags=re.MULTILINE,
    ).strip()
