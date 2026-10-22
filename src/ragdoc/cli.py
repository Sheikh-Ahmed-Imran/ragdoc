"""Command-line entry point: ``diagnose``, ``compare`` and ``batch`` against one PDF.

A thin wiring layer. It builds a :class:`RunConfig`, runs the stages in order
(loader -> chunker -> indexer -> retriever -> engine -> reporter) and prints the
report. Attribution, scoring and formatting all live in the components it calls;
nothing in the engine imports this module (``.ai/constitution.md``).

``diagnose`` takes one question; ``compare`` takes none; ``batch`` takes one question
set. ``diagnose`` and ``compare`` take the expected answer as an argument or from
``--answer-file``.
One PDF and one configuration per invocation.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from ragdoc.attribution.verdict import build_engine, compose_verdict
from ragdoc.batch import load_question_set
from ragdoc.batch import run_batch as execute_batch
from ragdoc.chunker import Chunker
from ragdoc.compare import compare_loaders
from ragdoc.config import RunConfig
from ragdoc.indexer.local_embeddings import LocalEmbeddingsIndexer
from ragdoc.loader.pymupdf_loader import PyMuPDFLoader
from ragdoc.loader import names as loader_names
from ragdoc.report import render_report
from ragdoc.report.batch import batch_to_json, render_batch_report
from ragdoc.report.compare import render_comparison
from ragdoc.retriever import Retriever

__all__ = ["build_parser", "main", "run", "run_batch", "run_compare"]

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


def _add_answer(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("expected_answer", nargs="?", default=None,
                        help="the answer text expected to be found in the PDF "
                             "(or use --answer-file)")
    parser.add_argument("--answer-file", default=None, metavar="PATH",
                        help="read the expected answer from this UTF-8 file "
                             "instead of the command line")


def _resolve_answer(args: argparse.Namespace) -> str:
    positional = args.expected_answer
    path = args.answer_file
    if positional is not None and path is not None:
        raise ValueError("give the expected answer either as an argument or with "
                         "--answer-file, not both")
    if path is not None:
        try:
            with open(path, encoding="utf-8") as handle:
                return handle.read().rstrip()
        except FileNotFoundError:
            raise FileNotFoundError(f"answer file not found: {path}") from None
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"cannot read answer file {path}: {exc}") from None
    if positional is None:
        raise ValueError("no expected answer given: pass it as an argument or "
                         "with --answer-file")
    return positional


def _add_config_flags(parser: argparse.ArgumentParser) -> None:
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


def _add_verbose(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="also print the evidence behind each verdict")


def _config_from(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        prepend_section_context=args.prepend_section_context,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragdoc",
        description="Diagnose which pipeline stage lost the answer to one question.",
    )
    sub = parser.add_subparsers(dest="command", metavar="{diagnose,compare,batch}")

    diagnose = sub.add_parser(
        "diagnose", help="diagnose which pipeline stage lost the answer",
    )
    diagnose.add_argument("pdf", help="path to the PDF")
    diagnose.add_argument("question", help="the question to ask the pipeline")
    _add_answer(diagnose)
    _add_config_flags(diagnose)
    _add_verbose(diagnose)

    compare = sub.add_parser(
        "compare", help="compare how each PDF loader extracts the expected answer",
    )
    compare.add_argument("pdf", help="path to the PDF")
    _add_answer(compare)
    compare.add_argument("--loaders", type=_loader_list, default=None, metavar="a,b",
                         help="comma-separated loader names (default: all registered)")
    _add_verbose(compare)

    batch = sub.add_parser(
        "batch", help="run a question set against one PDF and print the failure distribution",
    )
    batch.add_argument("pdf", help="path to the PDF")
    batch.add_argument("questions", help="path to the JSON question set")
    _add_config_flags(batch)
    batch.add_argument("--json", dest="json_path", default=None, metavar="PATH",
                       help="also write the structured result to this JSON file")
    return parser


def run(args: argparse.Namespace) -> str:
    expected_answer = _resolve_answer(args)
    config = _config_from(args)
    doc = PyMuPDFLoader().load(args.pdf)
    chunks = Chunker(config).chunk(doc)
    indexer = LocalEmbeddingsIndexer(config)
    indexer.index(chunks)
    ranking = Retriever(indexer, config).rank(args.question)
    result = build_engine().run(
        expected_answer=expected_answer,
        doc=doc,
        chunks=chunks,
        ranking=ranking,
        config=config,
    )
    return render_report(compose_verdict(result), config, verbose=args.verbose)


def run_compare(args: argparse.Namespace) -> str:
    return render_comparison(
        compare_loaders(args.pdf, _resolve_answer(args), args.loaders),
        verbose=args.verbose,
    )


def run_batch(args: argparse.Namespace) -> str:
    config = _config_from(args)
    pairs = load_question_set(args.questions)
    result = execute_batch(args.pdf, config, pairs)
    if args.json_path:
        payload = json.dumps(batch_to_json(result), indent=2, ensure_ascii=False)
        with open(args.json_path, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    return render_batch_report(result)


_COMMANDS = {"diagnose": run, "compare": run_compare, "batch": run_batch}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return 2
    try:
        report = _COMMANDS[args.command](args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ragdoc: error: {exc}", file=sys.stderr)
        return 2
    # The report is non-ASCII; a bare print fails on a cp1252 console.
    sys.stdout.buffer.write((report + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
