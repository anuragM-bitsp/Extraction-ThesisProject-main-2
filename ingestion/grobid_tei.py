"""
GROBID TEI-XML parsing.

Deliberately separated from grobid_client.py: this module has zero network
code, so it can be fully unit-tested against a canned TEI string without a
live GROBID server (there isn't one in this environment — GROBID normally
runs as its own Docker service). grobid_client.py is the thin, largely
untested-by-necessity layer that gets bytes onto the wire and TEI XML back;
this module is where the actual parsing logic (and its test coverage) lives.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from pydantic import BaseModel

TEI_NS = "{http://www.tei-c.org/ns/1.0}"


class GrobidSection(BaseModel):
    heading: str | None = None
    text: str = ""


class GrobidMetadata(BaseModel):
    title: str | None = None
    authors: list[str] = []
    abstract: str | None = None
    sections: list[GrobidSection] = []
    references: list[str] = []


def _text(el: ET.Element | None) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def parse_tei(tei_xml: str) -> GrobidMetadata:
    root = ET.fromstring(tei_xml)

    title = _text(root.find(f".//{TEI_NS}titleStmt/{TEI_NS}title")) or None

    authors: list[str] = []
    for pers in root.findall(f".//{TEI_NS}sourceDesc//{TEI_NS}author/{TEI_NS}persName"):
        forename = _text(pers.find(f"{TEI_NS}forename"))
        surname = _text(pers.find(f"{TEI_NS}surname"))
        name = " ".join(p for p in (forename, surname) if p)
        if name:
            authors.append(name)

    abstract = _text(root.find(f".//{TEI_NS}profileDesc/{TEI_NS}abstract")) or None

    sections: list[GrobidSection] = []
    for div in root.findall(f".//{TEI_NS}text/{TEI_NS}body/{TEI_NS}div"):
        heading = _text(div.find(f"{TEI_NS}head")) or None
        paragraphs = [_text(p) for p in div.findall(f"{TEI_NS}p")]
        text = "\n".join(p for p in paragraphs if p)
        if heading or text:
            sections.append(GrobidSection(heading=heading, text=text))

    references: list[str] = []
    for bibl in root.findall(f".//{TEI_NS}back//{TEI_NS}listBibl/{TEI_NS}biblStruct"):
        ref_title = _text(bibl.find(f".//{TEI_NS}title"))
        if ref_title:
            references.append(ref_title)

    return GrobidMetadata(
        title=title, authors=authors, abstract=abstract, sections=sections, references=references
    )
