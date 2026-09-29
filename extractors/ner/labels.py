from __future__ import annotations

from enum import Enum


class EntityLabel(str, Enum):
    PRECURSOR = "PRECURSOR"
    SOLVENT = "SOLVENT"
    MATERIAL = "MATERIAL"
    CHARACTERIZATION = "CHARACTERIZATION"
