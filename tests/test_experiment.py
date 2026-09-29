from __future__ import annotations

import pytest

from experiment.config import ExperimentConfig
from experiment.factory import build_extractor
from experiment.runner import ExperimentRunner
from experiment.tracking import MlflowTracker
from extractors.ner_extractor import NerExtractor
from extractors.hybrid_extractor import HybridExtractor
from extractors.llm.llm_client import FakeLLMClient
from extractors.llm_extractor import PromptLlmExtractor
from extractors.rule_extractor import RuleExtractor
from ingestion.canonical import BlockType, CanonicalBlock, CanonicalDocument
from schemas.extraction_schema import ExtractionResult, ExtractorName, Quantity, SynthesisExtraction


def block(text, page=1, section=None):
    return CanonicalBlock(block_type=BlockType.PARAGRAPH, page_number=page, section=section, text=text)


def _gold(paper_id, **fields) -> ExtractionResult:
    return ExtractionResult(
        paper_id=paper_id, version=1, extractor=ExtractorName.HUMAN, extractor_version="gold@test",
        prediction=SynthesisExtraction(paper_id=paper_id, **fields),
    )


# ---- ExperimentConfig ---------------------------------------------------------------


def test_as_params_returns_all_string_values():
    config = ExperimentConfig(name="exp1", extractor="rule", max_retries=5)
    params = config.as_params()
    assert all(isinstance(v, str) for v in params.values())
    assert params["max_retries"] == "5"
    assert params["extractor"] == "rule"


# ---- build_extractor ------------------------------------------------------------------


def test_build_extractor_rule_and_ner_need_no_dependencies():
    assert isinstance(build_extractor(ExperimentConfig(name="e", extractor="rule")), RuleExtractor)
    assert isinstance(build_extractor(ExperimentConfig(name="e", extractor="ner")), NerExtractor)


def test_build_extractor_llm_requires_llm_client():
    for kind in ("llm", "rag_llm", "hybrid"):
        with pytest.raises(ValueError):
            build_extractor(ExperimentConfig(name="e", extractor=kind))


def test_build_extractor_llm_and_hybrid_wire_dependencies_through():
    config = ExperimentConfig(name="e", extractor="hybrid", max_retries=7, prompt="Return {} only.", prompt_name="tiny")
    extractor = build_extractor(config, llm_client=FakeLLMClient([]))
    assert isinstance(extractor, HybridExtractor)
    assert extractor.llm_extractor.max_retries == 7
    assert extractor.llm_extractor.prompt == "Return {} only."
    assert extractor.llm_extractor.prompt_name == "tiny"

    legacy = build_extractor(ExperimentConfig(name="e", extractor="rag_llm"), llm_client=FakeLLMClient([]))
    assert isinstance(legacy, PromptLlmExtractor)


def test_experiment_params_log_prompt_hash_not_prompt_text():
    params = ExperimentConfig(name="e", extractor="llm", prompt="Extract things as JSON").as_params()
    assert "prompt" not in params
    assert len(params["prompt_sha"]) == 10


def test_build_extractor_unknown_kind_raises():
    with pytest.raises(ValueError):
        build_extractor(ExperimentConfig.model_construct(name="e", extractor="bogus"))


# ---- ExperimentRunner ------------------------------------------------------------------


def test_runner_scores_only_papers_with_gold():
    docs = [
        CanonicalDocument(paper_id="P001", version=1, blocks=[block("heated at 80 C for 2 h", section="Experimental")]),
        CanonicalDocument(paper_id="P002", version=1, blocks=[block("heated at 90 C for 1 h", section="Experimental")]),
    ]
    gold_by_paper = {"P001": _gold("P001", temperature=Quantity(value=80, unit="C"))}

    runner = ExperimentRunner(
        config=ExperimentConfig(name="e", extractor="rule"), extractor=RuleExtractor(), gold_by_paper=gold_by_paper
    )
    result = runner.run(docs)

    assert result.paper_count == 2
    assert result.scored_paper_count == 1
    assert result.ablation  # non-empty since one paper was scored


def test_runner_parallel_execution_matches_sequential():
    docs = [
        CanonicalDocument(paper_id=f"P{i:03d}", version=1, blocks=[block(f"heated at {80+i} C for 2 h")])
        for i in range(6)
    ]
    runner = ExperimentRunner(config=ExperimentConfig(name="e", extractor="rule"), extractor=RuleExtractor())

    sequential = runner.run(docs, max_workers=1)
    parallel = runner.run(docs, max_workers=4)

    seq_temps = [p.prediction.temperature.value for p in sequential.predictions]
    par_temps = [p.prediction.temperature.value for p in parallel.predictions]
    assert seq_temps == par_temps  # same order, same results


def test_runner_with_no_gold_produces_empty_ablation():
    docs = [CanonicalDocument(paper_id="P001", version=1, blocks=[block("heated at 80 C for 2 h")])]
    runner = ExperimentRunner(config=ExperimentConfig(name="e", extractor="rule"), extractor=RuleExtractor())
    result = runner.run(docs)
    assert result.scored_paper_count == 0
    assert result.ablation == []


def test_runner_works_end_to_end_with_hybrid_extractor():
    docs = [CanonicalDocument(
        paper_id="P001", version=1,
        blocks=[block("Silver nitrate (10 mM) was dissolved in ethanol and heated at 80 C for 2 h.", section="Experimental")],
    )]
    scripted = [{"synthesis_method": None, "steps": [], "precursors": [], "properties": []}]
    extractor = build_extractor(ExperimentConfig(name="e", extractor="hybrid"), llm_client=FakeLLMClient(scripted))
    runner = ExperimentRunner(config=ExperimentConfig(name="e", extractor="hybrid"), extractor=extractor)
    result = runner.run(docs)
    assert result.predictions[0].prediction.temperature.value == 80.0


# ---- MlflowTracker (real, offline, sqlite-backed) --------------------------------------


def test_mlflow_tracker_persists_params_and_metrics(tmp_path):
    docs = [CanonicalDocument(paper_id="P001", version=1, blocks=[block("heated at 80 C for 2 h")])]
    gold_by_paper = {"P001": _gold("P001", temperature=Quantity(value=80, unit="C"))}
    config = ExperimentConfig(name="rule-baseline", extractor="rule", max_retries=3)
    runner = ExperimentRunner(config=config, extractor=RuleExtractor(), gold_by_paper=gold_by_paper)
    result = runner.run(docs)

    db_path = tmp_path / "mlflow.db"
    tracker = MlflowTracker(tracking_uri=f"sqlite:///{db_path}", experiment_name="test-experiment")
    run_id = tracker.log_experiment_result(result)

    assert db_path.exists()

    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=f"sqlite:///{db_path}")
    run_data = client.get_run(run_id)
    assert run_data.data.params["extractor"] == "rule"
    assert run_data.data.params["max_retries"] == "3"
    assert run_data.data.metrics["paper_count"] == 1.0
    assert "RULE.field.f1" in run_data.data.metrics

    artifacts = client.list_artifacts(run_id)
    assert any(a.path == "reports.json" for a in artifacts)
