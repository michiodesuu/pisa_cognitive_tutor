"""
test_json_extractor.py
Tests that VLMExtractor._extract_json_list handles all messy model outputs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.ingestion.vlm_extractor import _extract_json_list, _validate_record


class TestExtractJsonList:
    def test_clean_json(self):
        raw = '[{"topic":"Heat","question_summary":"Q","correct_concept":"C","common_misconceptions":"M"}]'
        result = _extract_json_list(raw)
        assert len(result) == 1
        assert result[0]["topic"] == "Heat"

    def test_with_markdown_fence(self):
        raw = "```json\n[{\"topic\":\"Waves\",\"question_summary\":\"Q\",\"correct_concept\":\"C\",\"common_misconceptions\":\"M\"}]\n```"
        result = _extract_json_list(raw)
        assert len(result) == 1

    def test_with_preamble(self):
        raw = "Here is the extracted data:\n[{\"topic\":\"Energy\",\"question_summary\":\"Q\",\"correct_concept\":\"C\",\"common_misconceptions\":\"M\"}]\nThank you."
        result = _extract_json_list(raw)
        assert len(result) == 1

    def test_empty_array(self):
        result = _extract_json_list("[]")
        assert result == []

    def test_no_json(self):
        result = _extract_json_list("This page is a cover page.")
        assert result == []

    def test_multiple_records(self):
        raw = '[{"topic":"T1","question_summary":"Q1","correct_concept":"C1","common_misconceptions":"M1"},{"topic":"T2","question_summary":"Q2","correct_concept":"C2","common_misconceptions":"M2"}]'
        result = _extract_json_list(raw)
        assert len(result) == 2

    def test_nested_objects_preserved(self):
        raw = '[{"topic":"Genetics","question_summary":"Q","correct_concept":"C","common_misconceptions":"M","key_vocabulary":["DNA","RNA"]}]'
        result = _extract_json_list(raw)
        assert result[0]["key_vocabulary"] == ["DNA", "RNA"]


class TestValidateRecord:
    def _base(self):
        return {
            "topic": "Photosynthesis",
            "question_summary": "A leaf receives sunlight...",
            "correct_concept": "Light-dependent reactions",
            "common_misconceptions": "Plants get energy from soil",
            "pisa_competency": "Explain phenomena scientifically",
        }

    def test_valid_record(self):
        rec = self._base()
        result = _validate_record(rec, "test.pdf")
        assert result is not None
        assert result["source_file"] == "test.pdf"

    def test_missing_required_field(self):
        rec = self._base()
        del rec["correct_concept"]
        result = _validate_record(rec, "test.pdf")
        assert result is None

    def test_misconceptions_list_converted_to_string(self):
        rec = self._base()
        rec["common_misconceptions"] = ["Mistake 1", "Mistake 2"]
        result = _validate_record(rec, "test.pdf")
        assert isinstance(result["common_misconceptions"], str)
        assert ";" in result["common_misconceptions"]

    def test_defaults_filled(self):
        rec = self._base()
        result = _validate_record(rec, "test.pdf")
        assert result["pisa_knowledge_type"] == "Content"
        assert result["cognitive_demand"] == "Medium"
        assert result["key_vocabulary"] == []
