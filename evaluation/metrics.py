"""
PRF1: the one counting primitive entity-, relation-, and field-level
evaluation all build on (LLD section 22).
"""

from __future__ import annotations

from pydantic import BaseModel


class PRF1(BaseModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def __add__(self, other: "PRF1") -> "PRF1":
        return PRF1(tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn)
