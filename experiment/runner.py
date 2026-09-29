"""
ExperimentRunner (LLD section 23-24): the reproducible unit of "run this
extractor over this dataset and score it against gold."

    "I would make each paper independently processable... And P002 can
    process independently. That gives natural parallelism. For 50-100
    papers: Celery + Redis is more than sufficient. You don't need
    Kubernetes, Kafka and Spark just because this is called 'system
    design.'"

This runner IS that "each paper stands alone" property, expressed as plain
Python: `max_workers > 1` uses a thread pool with zero change to per-paper
logic. Threads specifically (not processes) because the real bottleneck for
llm/hybrid runs is I/O-bound LLM API calls, not CPU — exactly where
threads help despite the GIL. Distributing across Celery workers in
production is the same independence property moved to a different infra
layer, not a different execution model; nothing about this runner's logic
would need to change to make that swap.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from evaluation.ablation import ablation_table
from evaluation.evaluator import EvaluationReport, evaluate
from experiment.config import ExperimentConfig
from extractors.base import Extractor
from ingestion.canonical import CanonicalDocument
from schemas.extraction_schema import ExtractionResult


class ExperimentResult(BaseModel):
    config: ExperimentConfig
    predictions: list[ExtractionResult]
    reports: list[EvaluationReport]
    ablation: list[dict]

    @property
    def paper_count(self) -> int:
        return len(self.predictions)

    @property
    def scored_paper_count(self) -> int:
        """May be less than paper_count if gold isn't available for every
        paper yet — annotation (Step 10) doesn't have to be complete before
        an experiment can run, it just limits how many papers contribute to
        the reported metrics."""
        return len(self.reports)


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        extractor: Extractor,
        gold_by_paper: dict[str, ExtractionResult] | None = None,
    ):
        self.config = config
        self.extractor = extractor
        self.gold_by_paper = gold_by_paper or {}

    def run(self, documents: list[CanonicalDocument], max_workers: int = 1) -> ExperimentResult:
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                predictions = list(pool.map(self.extractor.extract, documents))
        else:
            predictions = [self.extractor.extract(doc) for doc in documents]

        reports = [
            evaluate(prediction, self.gold_by_paper[prediction.paper_id])
            for prediction in predictions
            if prediction.paper_id in self.gold_by_paper
        ]

        return ExperimentResult(
            config=self.config,
            predictions=predictions,
            reports=reports,
            ablation=ablation_table(reports),
        )
