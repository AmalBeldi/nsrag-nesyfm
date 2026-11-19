"""DGsumm: task-oriented graph summarization (paper Section 3.2 / Table 3).

Reduces the full patient graph DG_p to a task-relevant summary G^S_{p,T}
by (1) scoring nodes for relevance to the task/question and (2) pruning
low-relevance nodes and now-dangling edges, subject to a keep-ratio or
absolute size budget. This is what produces the "Before/After/Reduction"
statistics reported in Table 3 of the paper.

The relevance scorer is intentionally simple (lexical overlap +
recency) so it runs anywhere with no extra dependencies; swap in an
embedding-based scorer (e.g. a biomedical sentence encoder) for the
reported numbers if the paper's protocol specifies one -- record
whichever choice you make in `configs/default.yaml` for reproducibility.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from nsrag.schema import Edge, Node, PatientGraph

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


@dataclass
class DGsummConfig:
    keep_ratio: float = 0.25     # fraction of nodes to retain (paper reports ~75% reduction)
    min_nodes: int = 5           # never summarize below this many nodes
    recency_weight: float = 0.3  # weight given to how recent a node is
    relevance_weight: float = 0.7


class DGsumm:
    def __init__(self, config: DGsummConfig | None = None):
        self.config = config or DGsummConfig()

    def summarize(self, graph: PatientGraph, task_query: str) -> PatientGraph:
        query_terms = _tokenize(task_query)
        scored = self._score_nodes(graph, query_terms)

        n_keep = max(self.config.min_nodes, int(round(len(scored) * self.config.keep_ratio)))
        n_keep = min(n_keep, len(scored))
        kept_ids = {node_id for node_id, _ in scored[:n_keep]}

        summary = PatientGraph(patient_id=graph.patient_id)
        for node_id in kept_ids:
            summary.add_node(graph.nodes[node_id])
        for edge in graph.edges:
            if edge.subj in kept_ids and edge.obj in kept_ids:
                summary.add_edge(edge)

        return summary

    def _score_nodes(self, graph: PatientGraph, query_terms: set[str]) -> list[tuple[str, float]]:
        timestamps = sorted(
            {n.timestamp for n in graph.nodes.values() if n.timestamp}
        )
        rank_of = {ts: i for i, ts in enumerate(timestamps)}
        max_rank = max(len(timestamps) - 1, 1)

        scores: list[tuple[str, float]] = []
        for node_id, node in graph.nodes.items():
            node_terms = _tokenize(node.text)
            overlap = len(node_terms & query_terms)
            relevance = overlap / max(len(query_terms), 1)

            recency = 0.0
            if node.timestamp and node.timestamp in rank_of:
                recency = rank_of[node.timestamp] / max_rank

            score = (
                self.config.relevance_weight * relevance
                + self.config.recency_weight * recency
                + 1e-6  # keep a small positive floor so ties are stable
            )
            scores.append((node_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    @staticmethod
    def reduction_stats(before: PatientGraph, after: PatientGraph) -> dict:
        """Reproduces the columns of Table 3 (Average graph statistics
        before and after graph summarization) for a single patient
        graph; average these across the test set to fill the table."""
        def pct_reduction(b: int, a: int) -> float:
            return 0.0 if b == 0 else round(100.0 * (b - a) / b, 1)

        return {
            "nodes_before": before.n_nodes(),
            "nodes_after": after.n_nodes(),
            "nodes_reduction_pct": pct_reduction(before.n_nodes(), after.n_nodes()),
            "edges_before": before.n_edges(),
            "edges_after": after.n_edges(),
            "edges_reduction_pct": pct_reduction(before.n_edges(), after.n_edges()),
        }
