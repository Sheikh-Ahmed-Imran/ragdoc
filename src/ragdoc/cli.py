"""Command-line entry point: diagnose one question against one PDF.

A thin wiring layer. It builds a :class:`RunConfig`, runs the stages in order
(loader -> chunker -> indexer -> retriever -> engine -> reporter) and prints the
report. Attribution, scoring and formatting all live in the components it calls;
nothing in the engine imports this module (``.ai/constitution.md``).

One question per invocation; batch mode is out of scope for v0.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from ragdoc.attribution.verdict import build_engine, compose_verdict
from ragdoc.chunker import Chunker
from ragdoc.config import RunConfig
from ragdoc.indexer.local_embeddings import LocalEmbeddingsIndexer
from ragdoc.loader.pymupdf_loader import PyMuPDFLoader
from ragdoc.report import render_report
from ragdoc.retriever import Retriever

__all__ = ["build_parser", "main", "run"]

_DEFAULTS = RunConfig()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragdoc",
        description="Diagnose which pipeline stage lost the answer to one question.",
    )
    parser.add_argument("pdf", help="path to the PDF")
    parser.add_argument("question", help="the question to ask the pipeline")
    parser.add_argument("expected_answer", help="the answer text expected to be found in the PDF")
    parser.add_argument("--chunk-size", type=int, default=_DEFAULTS.chunk_size,
                        help="chunk size (default: %(default)s)")
    parser.add_argument("--overlap", type=int, default=_DEFAULTS.overlap,
                        help="chunk overlap (default: %(default)s)")
    parser.add_argument("--prepend-section-context",
                        action=argparse.BooleanOptionalAction,
                        default=_DEFAULTS.prepend_section_context,
                        help="prepend section path to chunk text before embedding "
                             "(default: %(default)s)")
    parser.add_argument("--embedding-model", default=_DEFAULTS.embedding_model,
                        help="embedding model (default: %(default)s)")
    parser.add_argument("--top-k", type=int, default=_DEFAULTS.top_k,
                        help="retrieval cutoff (default: %(default)s)")
    return parser


def run(args: argparse.Namespace) -> str:
    config = RunConfig(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        prepend_section_context=args.prepend_section_context,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
    )
    doc = PyMuPDFLoader().load(args.pdf)
    chunks = Chunker(config).chunk(doc)
    indexer = LocalEmbeddingsIndexer(config)
    indexer.index(chunks)
    ranking = Retriever(indexer, config).rank(args.question)
    result = build_engine().run(
        expected_answer=args.expected_answer,
        doc=doc,
        chunks=chunks,
        ranking=ranking,
        config=config,
    )
    return render_report(compose_verdict(result), config)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ragdoc: error: {exc}", file=sys.stderr)
        return 2
    # The report is non-ASCII; a bare print fails on a cp1252 console.
    sys.stdout.buffer.write((report + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
