"""
ExperimentConfig (LLD section 23).

    "Then use MLflow or similar experiment tracking to record: model
    version, prompt version, metrics, parameters, dataset version,
    experiment ID... Now you can reproduce: 'Hybrid-v3 achieved F1 = 0.87
    on Gold-v2.'"

That sentence only means something if "Hybrid-v3" refers to an exact,
recorded set of hyperparameters — not just a class name someone remembers
changing at some point. This config IS that recorded set; `as_params()` is
what gets logged to MLflow so a run can be reproduced later from its logged
parameters alone, without needing to read the code that produced it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ExtractorKind = Literal["rule", "ner", "llm", "rag_llm", "hybrid"]  # "rag_llm" = legacy alias of "llm"


class ExperimentConfig(BaseModel):
    name: str
    extractor: ExtractorKind

    schema_version: str = "v1"
    max_retries: int = 2

    # Logged for reproducibility even though the actual LLM client is
    # injected separately (see experiment/factory.py) — a model-name string
    # doesn't by itself know how to construct a client, and hard-coding that
    # mapping here would make this module responsible for every provider.
    llm_model_name: str = "unspecified"

    # Prompt + LLM: the extraction prompt used for this experiment. None ->
    # the default prompt (EXTRACTION_PROMPT_FILE or prompts/synthesis_extraction.txt).
    # The prompt's hash also ends up in the extractor version string.
    prompt: str | None = None
    prompt_name: str = "default"

    gold_dataset_version: str = "unspecified"
    notes: str = ""

    def as_params(self) -> dict[str, str]:
        """Flat str -> str dict, exactly what MLflow's log_params expects."""
        params = {k: str(v) for k, v in self.model_dump(exclude={"prompt"}).items()}
        if self.prompt:
            # Full prompt text can exceed MLflow's param length limit; log its
            # hash here (the text itself is in every result's llm_run.prompt).
            from extractors.llm.prompting import prompt_fingerprint

            params["prompt_sha"] = prompt_fingerprint(self.prompt)
        return params
