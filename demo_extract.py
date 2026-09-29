"""
Quick local demo — no database, no API server, no Redis required.

Runs ingestion (Step 3) + the Rule (Step 5) and NER (Step 6) extractors
on a single PDF and prints the structured result as JSON. These two
extractors need nothing beyond what `pip install -r requirements.txt`
already gives you.

If ANTHROPIC_API_KEY is set in your environment, it also runs the
Prompt + LLM extractor and the Hybrid extractor — those make real calls to
the Anthropic API and will incur API costs.

Usage:
    python3 demo_extract.py path/to/paper.pdf [path/to/prompt.txt]

The optional prompt file replaces the default extraction prompt
(prompts/synthesis_extraction.txt). LLM_MODEL selects the model.
"""

from __future__ import annotations

import json
import os
import sys

from extractors.hybrid_extractor import HybridExtractor
from extractors.ner_extractor import NerExtractor
from extractors.llm.llm_client import AnthropicLLMClient
from extractors.llm_extractor import PromptLlmExtractor
from orchestration.extractor_registry import configured_llm_model
from extractors.rule_extractor import RuleExtractor
from ingestion.pipeline import DocumentIngestionPipeline


def main(pdf_path: str, prompt_path: str | None = None) -> None:
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    print(f"Ingesting {pdf_path} ...")
    # No GROBID client passed -> falls back to PyMuPDF-only parsing (Step 3).
    # Pass grobid_client=HttpGrobidClient("http://localhost:8070") for
    # section-aware parsing + title/authors/abstract, if you have a GROBID
    # instance running (see README's Step 3 section for the docker command).
    pipeline = DocumentIngestionPipeline()
    doc = pipeline.ingest(paper_id="demo", version=1, pdf_bytes=pdf_bytes)
    print(f"Parsed {len(doc.blocks)} blocks (OCR ran: {doc.was_ocred}, GROBID used: {doc.used_grobid})\n")

    print("=== Rule-based extraction (Step 5) ===")
    rule_result = RuleExtractor().extract(doc)
    print(rule_result.prediction.model_dump_json(indent=2))

    print("\n=== NER extraction (Step 6) ===")
    ner_result = NerExtractor().extract(doc)
    print(ner_result.prediction.model_dump_json(indent=2))

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n(Set ANTHROPIC_API_KEY to also run the Prompt + LLM and Hybrid extractors — these make real API calls)")
        return

    prompt = open(prompt_path, encoding="utf-8").read() if prompt_path else None
    llm_client = AnthropicLLMClient(model=configured_llm_model(), api_key=api_key)
    llm_extractor = PromptLlmExtractor(llm_client=llm_client, prompt=prompt)

    print(f"\n=== Prompt + LLM extraction ({llm_client.model}) ===")
    llm_result = llm_extractor.extract(doc)
    print("Raw LLM JSON:")
    print(json.dumps(llm_result.llm_run.output, indent=2, ensure_ascii=False))
    print("\nMapped to the project schema:")
    print(llm_result.prediction.model_dump_json(indent=2))

    print("\n=== Hybrid extraction: Rule + NER + Prompt/LLM ===")
    # Reuses the LLM answer above instead of paying for a second API call.
    class _Cached(PromptLlmExtractor):
        def extract(self, document):
            return llm_result

    cached = _Cached(llm_client=llm_client, prompt=prompt)
    hybrid_result = HybridExtractor(llm_extractor=cached).extract(doc)
    print(hybrid_result.prediction.model_dump_json(indent=2))


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python3 demo_extract.py path/to/paper.pdf [path/to/prompt.txt]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
