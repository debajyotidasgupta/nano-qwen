"""Benchmark suite — runners for standard LLM and multimodal benchmarks.

Provides structured interfaces for running evaluations across:
  - Language: MMLU, GSM8K, MATH, HumanEval, ARC
  - Reasoning: AIME, GPQA
  - Multimodal: VQAv2, TextVQA, DocVQA
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Dict

from eval.evaluate import Evaluator


class BenchmarkRunner:
    """Run a suite of benchmarks and aggregate results."""

    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
        self.results: Dict[str, dict] = {}

    def run_benchmark(
        self,
        name: str,
        data_path: str,
        benchmark_type: str = "multiple_choice",
        max_samples: Optional[int] = None,
    ) -> dict:
        """Run a single benchmark.

        Args:
            name: Benchmark name (e.g., "mmlu", "gsm8k")
            data_path: Path to the benchmark data (JSONL)
            benchmark_type: "multiple_choice" or "generative"
            max_samples: Limit samples for quick eval
        """
        questions = self._load_data(data_path, max_samples)

        if benchmark_type == "multiple_choice":
            result = self.evaluator.eval_multiple_choice(questions)
        elif benchmark_type == "generative":
            result = self.evaluator.eval_generative(questions)
        else:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")

        self.results[name] = result
        print(f"[{name}] accuracy={result['accuracy']:.4f} ({result['correct']}/{result['total']})")
        return result

    def run_all(
        self,
        benchmarks: Dict[str, dict],
    ) -> Dict[str, dict]:
        """Run multiple benchmarks.

        Args:
            benchmarks: dict mapping name -> {"data_path": str, "type": str}
        """
        for name, cfg in benchmarks.items():
            self.run_benchmark(
                name=name,
                data_path=cfg["data_path"],
                benchmark_type=cfg.get("type", "multiple_choice"),
                max_samples=cfg.get("max_samples"),
            )
        return self.results

    def save_results(self, output_path: str):
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_path}")

    @staticmethod
    def _load_data(path: str, max_samples: Optional[int] = None) -> List[Dict]:
        """Load benchmark data from JSONL."""
        data = []
        p = Path(path)

        if not p.exists():
            print(f"Warning: benchmark data not found at {path}")
            return data

        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
                    if max_samples and len(data) >= max_samples:
                        break
        return data


# Pre-configured benchmark definitions
STANDARD_BENCHMARKS = {
    "mmlu": {"type": "multiple_choice", "data_path": "benchmarks/mmlu.jsonl"},
    "arc_challenge": {"type": "multiple_choice", "data_path": "benchmarks/arc_challenge.jsonl"},
    "hellaswag": {"type": "multiple_choice", "data_path": "benchmarks/hellaswag.jsonl"},
    "gsm8k": {"type": "generative", "data_path": "benchmarks/gsm8k.jsonl"},
    "math": {"type": "generative", "data_path": "benchmarks/math.jsonl"},
    "humaneval": {"type": "generative", "data_path": "benchmarks/humaneval.jsonl"},
}
