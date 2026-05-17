"""
dspy_optimizer.py
──────────────────
DSPy-based prompt optimizer for the cognitive assessment evaluator.

Optimizes the evaluator prompts against a baseline annotated dataset
to maximize inter-rater agreement (Fleiss' Kappa).

Strategy options:
  "bootstrap" → BootstrapFewShot (fast, ~10 min)
  "mipro"     → MIPROv2 (better quality, ~2h on CPU)

Usage:
  python -m src.evaluation.dspy_optimizer --config configs/model_configs.yaml
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)

# Dimension abbreviation for DSPy signature
DIM_NAMES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]


class CognitiveAssessmentSignature:
    """DSPy Signature: student response → cognitive dimension scores."""

    # These will be set dynamically by build_signature()
    pass


def build_dspy_signature():
    """Dynamically create the DSPy Signature class."""
    import dspy

    fields = {
        "pisa_context": dspy.InputField(desc="PISA question, correct concept, misconceptions"),
        "student_response": dspy.InputField(desc="Student's verbatim response"),
        "turn_number": dspy.InputField(desc="Conversation turn number (integer)"),
        "duration_sec": dspy.InputField(desc="Response time in seconds (float)"),
    }
    for dim in DIM_NAMES:
        fields[f"score_{dim.lower()}"] = dspy.OutputField(
            desc=f"Score for {dim} (0, 1, 2, or NA)"
        )
    fields["icap_level"] = dspy.OutputField(
        desc="Dominant ICAP level: Passive | Active | Constructive | Interactive"
    )
    fields["reasoning"] = dspy.OutputField(
        desc="1-2 sentence justification for the most notable scores"
    )

    return type("CognitiveDimensionAssessor", (dspy.Signature,), fields)


class DSPyOptimizer:
    """
    Wraps DSPy's BootstrapFewShot or MIPROv2 to optimize the evaluator prompt.

    The baseline dataset must be a JSONL file where each line has:
      {
        "pisa_context": "...",
        "student_response": "...",
        "turn_number": 3,
        "duration_sec": 22.4,
        "gold_C1": 1, "gold_C2": 2, ..., "gold_C8": 0,
        "gold_icap_level": "Constructive"
      }
    """

    def __init__(self, config_path: Path):
        self.cfg = yaml.safe_load(config_path.read_text())
        self.dspy_cfg = self.cfg["dspy"]
        self.config_path = config_path

    def _load_dataset(self) -> List[Dict[str, Any]]:
        dataset_path = Path(self.dspy_cfg["baseline_dataset_path"])
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Baseline dataset not found: {dataset_path}\n"
                "Create it by manually annotating ~50 student responses."
            )
        records = []
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        logger.info(f"[DSPy] Loaded {len(records)} baseline examples")
        return records

    def _records_to_examples(self, records: List[Dict]) -> list:
        """Convert JSONL records to DSPy Example objects."""
        import dspy
        examples = []
        for rec in records:
            ex = dspy.Example(
                pisa_context=rec.get("pisa_context", ""),
                student_response=rec.get("student_response", ""),
                turn_number=str(rec.get("turn_number", 1)),
                duration_sec=str(rec.get("duration_sec", 0.0)),
            )
            # Gold labels
            for dim in DIM_NAMES:
                ex[f"score_{dim.lower()}"] = str(rec.get(f"gold_{dim}", "NA"))
            ex["icap_level"] = rec.get("gold_icap_level", "Unknown")
            ex["reasoning"] = rec.get("gold_reasoning", "")
            examples.append(ex.with_inputs(
                "pisa_context", "student_response", "turn_number", "duration_sec"
            ))
        return examples

    def _kappa_metric(self, example, prediction, trace=None) -> float:
        """
        DSPy metric: simple dimension-level accuracy as a Kappa proxy.
        Full Kappa is computed post-hoc in reliability.py.
        """
        correct = 0
        total = 0
        for dim in DIM_NAMES:
            gold = str(example.get(f"score_{dim.lower()}", "NA"))
            pred = str(prediction.get(f"score_{dim.lower()}", "NA"))
            total += 1
            if gold == pred:
                correct += 1
        return correct / max(total, 1)

    def optimize(self):
        """Run DSPy optimization and save the compiled program."""
        import dspy

        # Configure DSPy LM (using the same Ollama backend)
        ollama_cfg = self.cfg["chatbot_llm"]
        lm = dspy.OllamaLocal(
            model=ollama_cfg["model_name"],
            base_url=ollama_cfg["ollama_base_url"],
            temperature=0.1,
            max_tokens=512,
        )
        dspy.configure(lm=lm)

        signature = build_dspy_signature()
        predictor = dspy.ChainOfThought(signature)

        records = self._load_dataset()
        if len(records) < 10:
            logger.error("[DSPy] Need at least 10 baseline examples. Aborting.")
            return

        examples = self._records_to_examples(records)
        train = examples[: int(len(examples) * 0.8)]
        dev = examples[int(len(examples) * 0.8) :]

        strategy = self.dspy_cfg.get("strategy", "bootstrap")

        if strategy == "mipro":
            optimizer = dspy.MIPROv2(
                metric=self._kappa_metric,
                num_candidates=self.dspy_cfg["num_candidates"],
                auto="light",
            )
            compiled = optimizer.compile(
                predictor, trainset=train, valset=dev,
                minibatch_size=min(25, len(train)),
            )
        else:
            optimizer = dspy.BootstrapFewShot(
                metric=self._kappa_metric,
                max_bootstrapped_demos=self.dspy_cfg["max_bootstrapped_demos"],
                max_labeled_demos=self.dspy_cfg["max_bootstrapped_demos"],
            )
            compiled = optimizer.compile(predictor, trainset=train)

        # Save compiled program
        out_dir = Path(self.dspy_cfg["optimized_prompts_path"])
        out_dir.mkdir(parents=True, exist_ok=True)
        compiled.save(str(out_dir / "compiled_assessor.json"))
        logger.info(f"[DSPy] Compiled program saved to {out_dir}")

        # Update evaluator_judge.txt with optimized few-shot demos
        self._export_few_shot_examples(compiled, out_dir)

    def _export_few_shot_examples(self, compiled, out_dir: Path):
        """Extract and save few-shot examples from compiled program."""
        try:
            demos = compiled.predict.demos
            export = [
                {
                    "pisa_context": d.pisa_context,
                    "student_response": d.student_response,
                    "scores": {dim: d.get(f"score_{dim.lower()}", "NA") for dim in DIM_NAMES},
                    "icap_level": d.icap_level,
                }
                for d in demos
            ]
            with (out_dir / "few_shot_demos.jsonl").open("w", encoding="utf-8") as f:
                for ex in export:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            logger.info(f"[DSPy] Exported {len(export)} few-shot demos")
        except Exception as e:
            logger.warning(f"[DSPy] Could not export demos: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_configs.yaml")
    args = parser.parse_args()
    optimizer = DSPyOptimizer(Path(args.config))
    optimizer.optimize()


if __name__ == "__main__":
    main()
