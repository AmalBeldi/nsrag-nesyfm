"""i2b2 2010 (concept/assertion/relation) loader.

The i2b2/n2c2 2010 challenge data is distributed by the n2c2 NLP
Research Data Sets portal under its own data use agreement:
https://n2c2.dbmi.hms.harvard.edu/2010-challenge

This loader parses the standard release layout: for each document
`<id>.txt` there are companion `<id>.con` (concepts), `<id>.ast`
(assertions), and `<id>.rel` (relations) annotation files using the
official i2b2 line format, e.g.:

    c="the mass" 5:3 5:4||t="problem"
    a="the mass" 5:3 5:4||t="problem"||v="present"
    c="the mass" 5:3 5:4||r="TeCP"||c="ct scan" 4:1 4:2

This gives component-level gold labels for Table 7 (entity/relation/
assertion/graph-matching evaluation), used precisely because it lets
RQ6 be measured independently of end-to-end MIMIC-III generation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_CONCEPT_RE = re.compile(r'c="(?P<text>[^"]*)" (?P<span>\S+ \S+)\|\|t="(?P<label>[^"]*)"')
_ASSERTION_RE = re.compile(
    r'a="(?P<text>[^"]*)" (?P<span>\S+ \S+)\|\|t="(?P<label>[^"]*)"\|\|v="(?P<value>[^"]*)"'
)
_RELATION_RE = re.compile(
    r'c="(?P<subj_text>[^"]*)" (?P<subj_span>\S+ \S+)\|\|r="(?P<rel>[^"]*)"'
    r'\|\|c="(?P<obj_text>[^"]*)" (?P<obj_span>\S+ \S+)'
)


@dataclass
class I2b2Concept:
    text: str
    span: str
    label: str


@dataclass
class I2b2Assertion:
    text: str
    span: str
    label: str
    value: str   # present | absent | possible | conditional | hypothetical | associated_with_someone_else


@dataclass
class I2b2Relation:
    subj_text: str
    subj_span: str
    relation: str
    obj_text: str
    obj_span: str


@dataclass
class I2b2Document:
    doc_id: str
    text: str
    concepts: list[I2b2Concept]
    assertions: list[I2b2Assertion]
    relations: list[I2b2Relation]


class I2b2Loader:
    def __init__(self, i2b2_dir: str | Path):
        self.dir = Path(i2b2_dir)

    def _read_lines(self, path: Path) -> list[str]:
        if not path.exists():
            return []
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()

    def load_document(self, doc_id: str) -> I2b2Document:
        txt_path = self.dir / f"{doc_id}.txt"
        if not txt_path.exists():
            raise FileNotFoundError(
                f"{txt_path} not found. Download the i2b2/n2c2 2010 release "
                "(DUA required) and point `i2b2_dir` at the extracted folder."
            )
        text = txt_path.read_text(encoding="utf-8", errors="ignore")

        concepts = [
            I2b2Concept(text=m["text"], span=m["span"], label=m["label"])
            for m in (mm.groupdict() for mm in _CONCEPT_RE.finditer(
                "\n".join(self._read_lines(self.dir / f"{doc_id}.con"))
            ))
        ]
        assertions = [
            I2b2Assertion(text=m["text"], span=m["span"], label=m["label"], value=m["value"])
            for m in (mm.groupdict() for mm in _ASSERTION_RE.finditer(
                "\n".join(self._read_lines(self.dir / f"{doc_id}.ast"))
            ))
        ]
        relations = [
            I2b2Relation(subj_text=m["subj_text"], subj_span=m["subj_span"], relation=m["rel"],
                         obj_text=m["obj_text"], obj_span=m["obj_span"])
            for m in (mm.groupdict() for mm in _RELATION_RE.finditer(
                "\n".join(self._read_lines(self.dir / f"{doc_id}.rel"))
            ))
        ]
        return I2b2Document(doc_id=doc_id, text=text, concepts=concepts,
                             assertions=assertions, relations=relations)

    def list_document_ids(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.txt"))
