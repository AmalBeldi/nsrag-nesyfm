"""B4 (GraphRAG) and B5 (LightRAG) baselines, Section 5.4/5.5.

The paper explicitly requires these to be *representative graph-based
RAG baselines*, i.e. the actual published systems, not re-implementations:

    B4: Microsoft GraphRAG    https://github.com/microsoft/graphrag
    B5: LightRAG (HKUDS)      https://github.com/HKUDS/LightRAG

Both ship their own indexing + retrieval + generation stack, so rather
than reimplementing their internals (which would not be a faithful
comparison and would undermine RQ4), these wrappers just adapt their
public Python APIs to the `Method` interface used by the rest of this
pipeline, so the *same* patient text corpus, the *same* question set,
and the *same* downstream factuality/verification evaluation
(Section 5.5: "For common factuality evaluation ... generated outputs
are processed by the same independent assertion-level evaluator")
apply uniformly across B1-B8.

Both `pip install graphrag` and `pip install lightrag-hku` need network
access and, for GraphRAG, an LLM API key for its indexing step (it
builds its own community-summary graph offline before querying) --
neither is available in this sandbox, so these wrappers are untested
here. Install and smoke-test them in your own environment; the
adapter logic (mapping patient notes in/answers out) is intentionally
minimal and should not need changes.
"""
from __future__ import annotations

from nsrag.baselines.common import Method
from nsrag.schema import GenerationOutput, timer


class GraphRAGBaseline(Method):
    """B4 -- GraphRAG (Edge et al., 2024).

    Requires: pip install graphrag
    GraphRAG indexes a corpus into community summaries ahead of time;
    `index_dir` should point at a directory already indexed via the
    `graphrag index` CLI over this patient's notes (one index per
    patient keeps the comparison patient-specific, matching Section 5.5:
    "Both systems operate on clinical information derived from the
    same patient records used by NS-RAG").
    """
    name = "B4 - GraphRAG"

    def __init__(self, generator, index_dir: str, search_type: str = "local"):
        super().__init__(generator)
        self.index_dir = index_dir
        self.search_type = search_type

    def run(self, patient_id, notes, structured_events, question, task) -> GenerationOutput:
        try:
            from graphrag.query.cli import run_local_search, run_global_search  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "GraphRAGBaseline requires the `graphrag` package "
                "(pip install graphrag) and a pre-built index at "
                "`index_dir` (see the module docstring)."
            ) from exc

        query_fn = run_local_search if self.search_type == "local" else run_global_search
        prompt_text = question or "Summarize this patient's clinical course."

        with timer() as t:
            answer = query_fn(self.index_dir, query=prompt_text)  # returns a string
        elapsed = t()

        return GenerationOutput(
            method=self.name, patient_id=patient_id, task=task, question=question,
            answer=str(answer), retrieved_context=[], context_tokens=0,
            retrieval_time_s=elapsed, generation_time_s=0.0,
            extra={"search_type": self.search_type, "index_dir": self.index_dir},
        )


class LightRAGBaseline(Method):
    """B5 -- LightRAG (Guo et al., 2024).

    Requires: pip install lightrag-hku
    `working_dir` should point at a directory already ingested via
    `LightRAG.insert(...)` over this patient's notes.
    """
    name = "B5 - LightRAG"

    def __init__(self, generator, working_dir: str, mode: str = "hybrid"):
        super().__init__(generator)
        self.working_dir = working_dir
        self.mode = mode

    def run(self, patient_id, notes, structured_events, question, task) -> GenerationOutput:
        try:
            from lightrag import LightRAG, QueryParam  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "LightRAGBaseline requires the `lightrag-hku` package "
                "(pip install lightrag-hku) and an ingested working "
                "directory at `working_dir` (see the module docstring)."
            ) from exc

        rag = LightRAG(working_dir=self.working_dir)
        prompt_text = question or "Summarize this patient's clinical course."

        with timer() as t:
            answer = rag.query(prompt_text, param=QueryParam(mode=self.mode))
        elapsed = t()

        return GenerationOutput(
            method=self.name, patient_id=patient_id, task=task, question=question,
            answer=str(answer), retrieved_context=[], context_tokens=0,
            retrieval_time_s=elapsed, generation_time_s=0.0,
            extra={"mode": self.mode, "working_dir": self.working_dir},
        )
