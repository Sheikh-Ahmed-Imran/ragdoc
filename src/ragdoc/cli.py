"""Command-line entry point: ``diagnose`` and ``compare`` one question against one PDF.

A thin wiring layer. It builds a :class:`RunConfig`, runs the stages in order
(loader -> chunker -> indexer -> retriever -> engine -> reporter) and prints the
report. Attribution, scoring and formatting all live in the components it calls;
nothing in the engine imports this module (``.ai/constitution.md``).

One question per invocation; batch mode is out of scope.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from ragdoc.attribution.verdict import build_engine, compose_verdict
from ragdoc.chunker import Chunker
from ragdoc.compare import compare_loaders
from ragdoc.config import RunConfig
from ragdoc.indexer.local_embeddings import LocalEmbeddingsIndexer
from ragdoc.loader.pymupdf_loader import PyMuPDFLoader
from ragdoc.loader import names as loader_names
from ragdoc.report import render_report
from ragdoc.report.compare import render_comparison
from ragdoc.retriever import Retriever

__all__ = ["build_parser", "main", "run", "run_compare"]

_DEFAULTS = RunConfig()


def _loader_list(value: str) -> list[str]:
    requested = [part.strip() for part in value.split(",") if part.strip()]
    if not requested:
        raise argparse.ArgumentTypeError("expected a comma-separated list of loader names")
    known = loader_names()
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown loader {', '.join(repr(n) for n in unknown)}; "
            f"registered loaders: {', '.join(known)}"
        )
    return requested


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("pdf", help="path to the PDF")
    parser.add_argument("question", help="the question to ask the pipeline")
    parser.add_argument("expected_answer", help="the answer text expected to be found in the PDF")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragdoc",
        description="Diagnose which pipeline stage lost the answer to one question.",
    )
    sub = parser.add_subparsers(dest="command", metavar="{diagnose,compare}")

    diagnose = sub.add_parser(
        "diagnose", help="diagnose which pipeline stage lost the answer",
    )
    _add_common(diagnose)
    diagnose.add_argument("--chunk-size", type=int, default=_DEFAULTS.chunk_size,
                          help="chunk size (default: %(default)s)")
    diagnose.add_argument("--overlap", type=int, default=_DEFAULTS.overlap,
                          help="chunk overlap (default: %(default)s)")
    diagnose.add_argument("--prepend-section-context",
                          action=argparse.BooleanOptionalAction,
                          default=_DEFAULTS.prepend_section_context,
                          help="prepend section path to chunk text before embedding "
                               "(default: %(default)s)")
    diagnose.add_argument("--embedding-model", default=_DEFAULTS.embedding_model,
                          help="embedding model (default: %(default)s)")
    diagnose.add_argument("--top-k", type=int, default=_DEFAULTS.top_k,
                          help="retrieval cutoff (default: %(default)s)")

    compare = sub.add_parser(
        "compare", help="compare how each PDF loader extracts the expected answer",
    )
    _add_common(compare)
    compare.add_argument("--loaders", type=_loader_list, default=None, metavar="a,b",
                         help="comma-separated loader names (default: all registered)")
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


def run_compare(args: argparse.Namespace) -> str:
    return render_comparison(compare_loaders(args.pdf, args.expected_answer, args.loaders))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return 2
    try:
        report = run_compare(args) if args.command == "compare" else run(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ragdoc: error: {exc}", file=sys.stderr)
        return 2
    # The report is non-ASCII; a bare print fails on a cp1252 console.
    sys.stdout.buffer.write((report + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
