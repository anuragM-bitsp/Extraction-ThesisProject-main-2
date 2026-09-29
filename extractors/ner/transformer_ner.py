"""
Production NER model: a fine-tuned domain transformer via Hugging Face
`transformers` (LLD section 12 — "fine-tuning a domain model on our
manually annotated data rather than relying purely on generic NER").

Not exercised by the test suite: loading any pretrained or fine-tuned
weights requires Hugging Face Hub access, which this sandboxed environment
doesn't have. Swap this in for GazetteerNER once deployed somewhere with
model access — both implement `NerModel`, so nothing in NerExtractor or the
linking logic changes.
"""

from __future__ import annotations

from extractors.ner.base import Entity, NerModel
from extractors.ner.labels import EntityLabel

# Map whatever label scheme the fine-tuned model was trained with onto this
# project's EntityLabel. Adjust to match the actual model's label set.
_HF_LABEL_MAP = {
    "PRECURSOR": EntityLabel.PRECURSOR,
    "SOLVENT": EntityLabel.SOLVENT,
    "MATERIAL": EntityLabel.MATERIAL,
    "CHAR_METHOD": EntityLabel.CHARACTERIZATION,
}


class TransformerNER(NerModel):
    def __init__(self, model_name: str = "path/to/fine-tuned-materials-ner"):
        from transformers import pipeline  # local import: optional heavy dep

        self._pipeline = pipeline("ner", model=model_name, aggregation_strategy="simple")
        self.name = model_name

    def predict(self, text: str) -> list[Entity]:
        entities: list[Entity] = []
        for raw in self._pipeline(text):
            label = _HF_LABEL_MAP.get(raw["entity_group"])
            if label is None:
                continue  # model predicted a label outside our schema (e.g. O, MISC)
            entities.append(
                Entity(
                    text=raw["word"],
                    label=label,
                    char_start=raw["start"],
                    char_end=raw["end"],
                    confidence=float(raw["score"]),
                )
            )
        return entities
