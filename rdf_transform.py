"""HERSE: RDF transformation of the patient graph (paper Section 3.2).

Converts a `PatientGraph` (nodes + edges) into a list of `Triple`
objects (G^RDF_{p,T}). A real `rdflib.Graph` is used when available so
you get actual serialization (Turtle/N-Triples) for archival and for
feeding SPARQL-based verification; a lightweight in-memory fallback
(pure `Triple` list) is used otherwise so the rest of the pipeline
never hard-depends on rdflib being installed.
"""
from __future__ import annotations

from nsrag.schema import PatientGraph, Triple

_NS = "urn:nsrag:"


def _uri(s: str) -> str:
    return f"{_NS}{s}"


class HERSE:
    """Handling and Enhancing RDF Summarization (blank-node elimination
    per the paper's citation [5]): every node gets a stable, dense URI
    instead of an RDF blank node, which keeps the graph mergeable across
    summarization runs and avoids blank-node explosion.
    """

    def to_triples(self, graph: PatientGraph) -> list[Triple]:
        triples: list[Triple] = []
        for node in graph.nodes.values():
            triples.append(Triple(subj=_uri(node.id), pred="rdf:type", obj=node.label,
                                   provenance=node.id))
            triples.append(Triple(subj=_uri(node.id), pred="hasText", obj=node.text,
                                   provenance=node.id))
            if node.timestamp:
                triples.append(Triple(subj=_uri(node.id), pred="hasTimestamp",
                                       obj=node.timestamp, provenance=node.id))
            for k, v in node.attrs.items():
                if v is not None:
                    triples.append(Triple(subj=_uri(node.id), pred=f"has{k.capitalize()}",
                                           obj=str(v), provenance=node.id))

        for edge in graph.edges:
            triples.append(Triple(subj=_uri(edge.subj), pred=edge.pred, obj=_uri(edge.obj),
                                   provenance=f"{edge.subj}->{edge.obj}"))
        return triples

    def to_rdflib_graph(self, graph: PatientGraph):
        """Optional real-RDF export. Requires `pip install rdflib`."""
        try:
            import rdflib  # type: ignore
            from rdflib import URIRef, Literal, Namespace  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "to_rdflib_graph() requires rdflib: pip install rdflib"
            ) from exc

        NS = Namespace(_NS)
        g = rdflib.Graph()
        g.bind("nsrag", NS)
        for triple in self.to_triples(graph):
            s = URIRef(triple.subj) if triple.subj.startswith(_NS) else Literal(triple.subj)
            o = URIRef(triple.obj) if triple.obj.startswith(_NS) else Literal(triple.obj)
            g.add((s, NS[triple.pred], o))
        return g
