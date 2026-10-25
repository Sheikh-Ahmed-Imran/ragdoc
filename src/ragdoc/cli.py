"""Command-line entry point: ``diagnose``, ``compare``, ``batch`` and ``sweep`` against one PDF.

A thin wiring layer. It builds a :class:`RunConfig`, runs the stages in order
(loader -> chunker -> indexer -> retriever -> engine -> reporter) and prints the
report. Attribution, scoring and formatting all live in the components it calls;
nothing in the engine imports this module (``.ai/constitution.md``).

``diagnose`` takes one question; ``compare`` takes none; ``batch`` takes one question
set; ``sweep`` takes a question set and a list of values per knob. ``diagnose`` and ``compare`` take the expected answer as an argument or from
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
from ragdoc.loader import get as get_loader
from ragdoc.loader import names as loader_names
from ragdoc.report import render_report
from ragdoc.report.batch import batch_to_json, render_batch_report
from ragdoc.report.compare import render_comparison
from ragdoc.report.retrieved import render_retrieved
from ragdoc.report.sweep import OBJECTIVES, render_sweep_table, sweep_to_json
from ragdoc.report.symbols import Symbols, resolve
from ragdoc.retriever import Retriever
from ragdoc.sweep import build_grid, count_forward_passes
from ragdoc.sweep import run_sweep as execute_sweep

__all__ = ["build_parser", "main", "run", "run_batch", "run_compare", "run_sweep"]

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


def _loader_name(value: str) -> str:
    known = loader_names()
    if value not in known:
        raise argparse.ArgumentTypeError(
            f"unknown loader {value!r}; registered loaders: {', '.join(known)}"
        )
    return value


def _csv(value: str, convert, what: str) -> list:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(f"expected a comma-separated list of {what}")
    try:
        return [convert(part) for part in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated list of {what}, got {value!r}"
        ) from None


def _int_list(value: str) -> list[int]:
    return _csv(value, int, "integers")


def _str_list(value: str) -> list[str]:
    return _csv(value, str, "names")


def _parse_bool(text: str) -> bool:
    lowered = text.lower()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no"):
        return False
    raise ValueError(text)


def _bool_list(value: str) -> list[bool]:
    return _csv(value, _parse_bool, "true/false values")


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
            # utf-8-sig drops a leading BOM (added by some Windows editors) so it
            # cannot become part of the expected answer.
            with open(path, encoding="utf-8-sig") as handle:
                resolved = handle.read().rstrip()
        except FileNotFoundError:
            raise FileNotFoundError(f"answer file not found: {path}") from None
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"cannot read answer file {path}: {exc}") from None
    elif positional is None:
        raise ValueError("no expected answer given: pass it as an argument or "
                         "with --answer-file")
    else:
        resolved = positional
    if not resolved.strip():
        raise ValueError("the expected answer is empty: there is nothing to "
                         "locate in the document")
    return resolved


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
    parser.add_argument("--loader", type=_loader_name, default=_DEFAULTS.loader,
                        metavar="NAME",
                        help="PDF loader to extract with; one of the registered "
                             "loaders (default: %(default)s)")


def _add_verbose(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="also print the evidence behind each verdict")


def _add_ascii(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ascii", action="store_true",
                        help="draw status symbols in plain ASCII (automatic when the "
                             "console cannot show Unicode)")


def _symbols_for(args: argparse.Namespace) -> Symbols:
    # Display only: never reaches RunConfig or the engine.
    return resolve(sys.stdout, force_ascii=args.ascii)


def _config_from(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        prepend_section_context=args.prepend_section_context,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        loader=args.loader,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragdoc",
        description="Diagnose which pipeline stage lost the answer to one question.",
    )
    sub = parser.add_subparsers(dest="command", metavar="{diagnose,compare,batch,sweep}")

    diagnose = sub.add_parser(
        "diagnose", help="diagnose which pipeline stage lost the answer",
    )
    diagnose.add_argument("pdf", help="path to the PDF")
    diagnose.add_argument("question", help="the question to ask the pipeline")
    _add_answer(diagnose)
    _add_config_flags(diagnose)
    _add_verbose(diagnose)
    _add_ascii(diagnose)
    diagnose.add_argument("--show-retrieved", action="store_true",
                          help="also list the top-k chunks the retriever returned, "
                               "marking the chunk that holds your answer")

    compare = sub.add_parser(
        "compare", help="compare how each PDF loader extracts the expected answer",
    )
    compare.add_argument("pdf", help="path to the PDF")
    _add_answer(compare)
    compare.add_argument("--loaders", type=_loader_list, default=None, metavar="a,b",
                         help="comma-separated loader names (default: all registered)")
    _add_verbose(compare)
    _add_ascii(compare)

    batch = sub.add_parser(
        "batch", help="run a question set against one PDF and print the failure distribution",
    )
    batch.add_argument("pdf", help="path to the PDF")
    batch.add_argument("questions", help="path to the JSON question set")
    _add_config_flags(batch)
    _add_ascii(batch)
    batch.add_argument("--json", dest="json_path", default=None, metavar="PATH",
                       help="also write the structured result to this JSON file")

    sweep = sub.add_parser(
        "sweep", help="run a question set across a grid of configs and name the best one",
    )
    sweep.add_argument("pdf", help="path to the PDF")
    sweep.add_argument("questions", help="path to the JSON question set")
    sweep.add_argument("--chunk-size", type=_int_list, default=None, metavar="a,b",
                       help=f"comma-separated chunk sizes (default: {_DEFAULTS.chunk_size})")
    sweep.add_argument("--overlap", type=_int_list, default=None, metavar="a,b",
                       help=f"comma-separated overlaps (default: {_DEFAULTS.overlap})")
    sweep.add_argument("--top-k", type=_int_list, default=None, metavar="a,b",
                       help=f"comma-separated retrieval cutoffs (default: {_DEFAULTS.top_k})")
    sweep.add_argument("--embedding-model", type=_str_list, default=None, metavar="a,b",
                       help="comma-separated embedding models "
                            f"(default: {_DEFAULTS.embedding_model})")
    sweep.add_argument("--prepend-section-context", type=_bool_list, default=None,
                       metavar="true,false",
                       help="comma-separated true/false values "
                            f"(default: {str(_DEFAULTS.prepend_section_context).lower()})")
    sweep.add_argument("--objective", choices=OBJECTIVES, default="no-failure",
                       help="what the winner maximises; MRR breaks ties "
                            "(default: %(default)s)")
    sweep.add_argument("--json", dest="json_path", default=None, metavar="PATH",
                       help="also write the full per-config results to this JSON file")
    _add_ascii(sweep)
    return parser


def run(args: argparse.Namespace) -> str:
    expected_answer = _resolve_answer(args)
    config = _config_from(args)
    doc = get_loader(config.loader).load(args.pdf)
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
    verdict = compose_verdict(result)
    symbols = _symbols_for(args)
    report = render_report(verdict, config, verbose=args.verbose, symbols=symbols)
    if args.show_retrieved:
        target_id = _target_chunk_id(verdict)
        report += "\n\n" + render_retrieved(
            ranking, chunks, target_id, config.top_k, symbols=symbols
        )
    return report


def _target_chunk_id(verdict) -> Optional[str]:
    # Read the raw check result, not the displayed status: the missing-context
    # case displays chunking as FAIL while survive still passed with a target.
    for stage in verdict.stages:
        if stage.stage == "chunking":
            result = stage.result
            if result is not None and result.is_pass:
                return result.evidence.get("target_chunk_id")
    return None


def run_compare(args: argparse.Namespace) -> str:
    return render_comparison(
        compare_loaders(args.pdf, _resolve_answer(args), args.loaders),
        verbose=args.verbose,
        symbols=_symbols_for(args),
    )


def run_batch(args: argparse.Namespace) -> str:
    config = _config_from(args)
    pairs = load_question_set(args.questions)
    result = execute_batch(args.pdf, config, pairs)
    if args.json_path:
        payload = json.dumps(batch_to_json(result), indent=2, ensure_ascii=False)
        with open(args.json_path, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    return render_batch_report(result, _symbols_for(args))


def _emit(text: str) -> None:
    # Non-ASCII safe, like main's report write.
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def run_sweep(args: argparse.Namespace) -> str:
    knobs = dict(
        chunk_sizes=args.chunk_size,
        overlaps=args.overlap,
        top_ks=args.top_k,
        embedding_models=args.embedding_model,
        prepend_section_context=args.prepend_section_context,
    )
    pairs = load_question_set(args.questions)
    grid = build_grid(**knobs)
    passes = count_forward_passes(grid)
    _emit(f"Sweep: {len(grid)} config{'s' if len(grid) != 1 else ''}, "
          f"{passes} forward pass{'es' if passes != 1 else ''}")

    def progress(done: int, total: int, config: RunConfig) -> None:
        print(f"[{done}/{total}] chunk_size={config.chunk_size} overlap={config.overlap} "
              f"top_k={config.top_k} model={config.embedding_model} "
              f"prepend={str(config.prepend_section_context).lower()}",
              file=sys.stderr, flush=True)

    result = execute_sweep(args.pdf, pairs, progress=progress, **knobs)
    if args.json_path:
        payload = json.dumps(sweep_to_json(result, args.objective), indent=2,
                             ensure_ascii=False)
        with open(args.json_path, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    return render_sweep_table(result, args.objective, _symbols_for(args))


_COMMANDS = {"diagnose": run, "compare": run_compare, "batch": run_batch,
             "sweep": run_sweep}


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
