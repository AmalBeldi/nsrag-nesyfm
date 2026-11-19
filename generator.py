"""Generation backend (paper Section 5.6: Flan-T5-Large, temp=0.2, 2048-token context).

Defines a small `Generator` interface with:

  * `HFGenerator`   -- real backend used to reproduce the paper's
                        numbers. Requires `torch` + `transformers` and
                        a GPU for reasonable latency (paper: A100 40GB).
  * `MockGenerator` -- deterministic, dependency-free stand-in used for
                        unit tests and for validating the rest of the
                        pipeline (retrieval -> prompting -> verification
                        -> metrics) without downloading any model
                        weights. It never produces meaningful clinical
                        text and must not be used to report results.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    generation_time_s: float
    prompt_tokens: int
    output_tokens: int


class Generator(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 256) -> GenerationResult:
        ...


PROMPT_TEMPLATE = (
    "You are a clinical assistant. Answer the question using ONLY the "
    "evidence provided below. If the evidence is insufficient, say so "
    "explicitly instead of guessing.\n\n"
    "Evidence:\n{evidence}\n\nQuestion: {question}\nAnswer:"
)


def build_prompt(evidence: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(evidence=evidence, question=question)


class HFGenerator(Generator):
    """Wraps a HuggingFace seq2seq model (default: Flan-T5-Large, as
    specified in Section 5.6 of the paper)."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        device: str = "cuda",
        temperature: float = 0.2,
    ):
        try:
            import torch  # type: ignore
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "HFGenerator requires torch and transformers:\n"
                "  pip install torch transformers\n"
                "Use MockGenerator for dependency-free pipeline testing."
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.device = device
        self.temperature = temperature

    def generate(self, prompt: str, max_new_tokens: int = 256) -> GenerationResult:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        start = time.perf_counter()
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-4),
            )
        elapsed = time.perf_counter() - start

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return GenerationResult(
            text=text,
            generation_time_s=elapsed,
            prompt_tokens=int(inputs["input_ids"].shape[-1]),
            output_tokens=int(output_ids.shape[-1]),
        )


class MockGenerator(Generator):
    """Deterministic stub for pipeline/unit testing ONLY.

    Concatenates a fixed number of evidence lines from the prompt into
    a templated "answer" so that downstream assertion extraction and
    verification have something well-formed to operate on. This must
    never be used to produce numbers reported in the paper.
    """

    def generate(self, prompt: str, max_new_tokens: int = 256) -> GenerationResult:
        start = time.perf_counter()
        evidence_block = prompt.split("Evidence:\n", 1)[-1].split("\n\nQuestion:")[0]
        lines = [l.strip("- ").strip() for l in evidence_block.splitlines() if l.strip()]
        answer = " ".join(lines[:3]) if lines else "No sufficient evidence was found."
        elapsed = time.perf_counter() - start
        return GenerationResult(
            text=answer,
            generation_time_s=elapsed,
            prompt_tokens=len(prompt.split()),
            output_tokens=len(answer.split()),
        )
