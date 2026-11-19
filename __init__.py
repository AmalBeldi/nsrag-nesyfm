"""NS-RAG: Graph-Grounded Neuro-Symbolic Retrieval-Augmented Generation.

Reference implementation of a pipeline for graph-grounded, three-state
verification of generated clinical answers.

This package is organized by pipeline stage:

    data          -> MIMIC-III / i2b2 2010 loaders
    graph         -> GraphSynth (construction), DGsumm (summarization),
                     HERSE (RDF transformation)
    retrieval     -> graph-grounded and textual retrieval
    baselines     -> B1-B7 comparison methods
    generation    -> LLM generation backend (HuggingFace)
    verification  -> three-state symbolic verification + explanation
    metrics       -> all metrics defined in Section 5.8 of the paper
    pipeline      -> end-to-end experiment runners (one per paper table)
    utils         -> results -> LaTeX table generation, timing, I/O
"""

__version__ = "0.1.0"
