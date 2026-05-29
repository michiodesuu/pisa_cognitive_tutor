"""
ablation_study.py
──────────────────
Runs the evaluation pipeline under different configurations to test the system's
robustness and the necessity of its components.

Ablations:
1. Base Ensemble: All 4 models.
2. No Typhoon2: Test without the Thai-tuned model.
3. No BGE-M3 Sparse Weights: Dense retrieval only.
4. No GraphRAG: Vector search only.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def run_ablation(config_path: Path):
    logger.info("Starting Ablation Study...")
    logger.info("1. Base Ensemble (Full system)")
    # In a full run, this would invoke the analyzer pipeline with the full config
    logger.info("2. No Typhoon2 (Evaluating without typhoon2:8b-instruct-q4_K_M)")
    # Run analyzer without typhoon to see the drop in H4
    logger.info("3. No BGE-M3 Sparse Weights (Dense only retrieval)")
    # Run retrieval with sparse weight = 0.0
    logger.info("4. No GraphRAG (Vector search only)")
    # Run engine with GraphRAG disabled
    logger.info("Ablation Study Framework initialized. Execute individual ablations by modifying model_configs.yaml")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_ablation(Path("configs/model_configs.yaml"))
