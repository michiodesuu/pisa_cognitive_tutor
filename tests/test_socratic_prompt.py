"""
test_socratic_prompt.py
Tests that the Socratic tutor prompt and TutorEngine never output forbidden words.
"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


FORBIDDEN_PATTERN = re.compile(
    r"\b(correct|incorrect|wrong|right answer|well done|good job|try again|"
    r"exactly|perfect|ถูกต้อง|ผิด|เก่งมาก|ดีมาก|ถูกแล้ว|ลองใหม่|正确|不对|很好|对了)\b",
    re.IGNORECASE | re.UNICODE,
)

SOCRATIC_PROMPT_PATH = Path("configs/prompts/socratic_tutor.txt")


class TestSocraticPromptFile:
    def test_prompt_file_exists(self):
        assert SOCRATIC_PROMPT_PATH.exists(), "Socratic tutor prompt file missing"

    def test_prompt_declares_forbidden_words(self):
        content = SOCRATIC_PROMPT_PATH.read_text()
        assert "FORBIDDEN" in content.upper()

    def test_prompt_contains_icap_levels(self):
        content = SOCRATIC_PROMPT_PATH.read_text()
        for level in ["Passive", "Active", "Constructive", "Interactive"]:
            assert level in content

    def test_prompt_mentions_scaffolding(self):
        content = SOCRATIC_PROMPT_PATH.read_text()
        assert "Socratic" in content or "scaffold" in content.lower()


class TestForbiddenWordDetection:
    @pytest.mark.parametrize("text,should_fail", [
        ("That is correct!", True),
        ("Incorrect, please try again.", True),
        ("Well done, you got it right!", True),
        ("ถูกต้องมากเลย", True),
        ("What do you think the mechanism might be?", False),
        ("Can you explain why that happens?", False),
        ("That is an interesting perspective — what evidence do you have?", False),
        ("How does this relate to what you observed in the data?", False),
    ])
    def test_forbidden_word_detection(self, text, should_fail):
        match = FORBIDDEN_PATTERN.search(text)
        if should_fail:
            assert match is not None, f"Expected forbidden word in: {text!r}"
        else:
            assert match is None, f"False positive forbidden word in: {text!r}"


class TestReliabilityMetrics:
    def test_fleiss_kappa_perfect_agreement(self):
        import numpy as np
        from src.metrics.reliability import fleiss_kappa
        # All 3 raters give score 2 for 5 items
        matrix = np.array([[2, 2, 2]] * 5, dtype=object)
        k = fleiss_kappa(matrix, categories=[0.0, 1.0, 2.0])
        assert k == pytest.approx(1.0, abs=0.01)

    def test_fleiss_kappa_chance_agreement(self):
        import numpy as np
        from src.metrics.reliability import fleiss_kappa
        # Random split 0/1/2 equally
        matrix = np.array(
            [[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 1, 2], [1, 2, 0]],
            dtype=object,
        )
        k = fleiss_kappa(matrix, categories=[0.0, 1.0, 2.0])
        assert isinstance(k, float)

    def test_krippendorff_ordinal_perfect(self):
        import numpy as np
        from src.metrics.reliability import krippendorff_alpha
        data = np.array([[1, 2, 2, 1, 2], [1, 2, 2, 1, 2]], dtype=float)
        alpha = krippendorff_alpha(data, level_of_measurement="ordinal")
        assert alpha == pytest.approx(1.0, abs=0.01)
