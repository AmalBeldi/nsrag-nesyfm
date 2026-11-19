"""CLI: python -m nsrag.scripts.run_i2b2_cli --config configs/default.yaml --out results/table7_i2b2.json

Uses `data.i2b2_dir` from the config file (no --instances needed: the
i2b2 release's own .txt/.con/.ast/.rel files are the instances)."""
from __future__ import annotations

import argparse

from nsrag.graph.extraction import RuleBasedExtractor
from nsrag.pipeline.run_i2b2_eval import run_i2b2_component_evaluation
from nsrag.scripts.common import load_config, write_json


def main():
    parser = argparse.ArgumentParser(description="Run i2b2 2010 component-level evaluation (Table 7).")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N documents (useful for a quick smoke test).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    extractor = RuleBasedExtractor()  # swap for SciSpacyExtractor to reproduce the paper's numbers

    from nsrag.data.i2b2_loader import I2b2Loader
    loader = I2b2Loader(config["data"]["i2b2_dir"])
    doc_ids = loader.list_document_ids()
    if args.limit:
        doc_ids = doc_ids[: args.limit]

    results = run_i2b2_component_evaluation(
        i2b2_dir=config["data"]["i2b2_dir"],
        entity_extractor=extractor, relation_extractor=extractor,
        doc_ids=doc_ids,
    )
    serializable = {
        name: {"precision": r.precision, "recall": r.recall, "f1": r.f1, "tp": r.tp, "fp": r.fp, "fn": r.fn}
        for name, r in results.items()
    }
    write_json(serializable, args.out)


if __name__ == "__main__":
    main()
