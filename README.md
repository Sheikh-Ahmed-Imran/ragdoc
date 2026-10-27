# ragdoc

[![PyPI](https://img.shields.io/pypi/v/ragdoc.svg?cacheSeconds=3600)](https://pypi.org/project/ragdoc/)
[![Python](https://img.shields.io/pypi/pyversions/ragdoc.svg?cacheSeconds=3600)](https://pypi.org/project/ragdoc/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Find out which stage of your PDF RAG pipeline lost the answer.**

When a retrieval-augmented pipeline fails to answer a question you *know* the
document contains, the hard part is rarely the fix — it is working out **where** it
broke. A bad chunk size and a bad retriever look identical from the outside: in both
cases the answer just doesn't come back. `ragdoc` ends that guesswork. Give it a PDF,
a question, and the answer you know is in the document, and it tells you the single
stage that lost it — **extraction**, **chunking**, or **retrieval** — with the
evidence behind the verdict.

It is a headless, offline, deterministic CLI. No API keys, no LLM calls, no telemetry.

## Project status

ragdoc is in **beta** (`0.x`). Its attribution engine and CLI are covered by an
extensive automated test suite — unit tests plus integration tests against fixture
PDFs that exercise each stage and every failure mode. What is still maturing is
*breadth*: validation across a wide range of real-world PDFs is ongoing, so on unusual
layouts or heavily structured documents a verdict may warrant a careful read. Treat the
output as strong, evidence-backed signal rather than an oracle, and expect the occasional
rough edge before `1.0`. Reports of PDFs it gets wrong are especially welcome.

## How it works

`ragdoc` runs your document through the forward pipeline once, then runs a chain of
backward checks that each consume one stage's output:

```
load → chunk → index → retrieve          (the forward pipeline)
  │       │       │         │
locate → survive ───────→ rank            (the backward checks)
```

- **locate** — is the expected answer actually in the extracted text? If not, the
  loader lost it → **extraction**.
- **survive** — does the answer survive chunking intact inside one chunk, with the
  section context it needs to be findable? If a boundary cut it, or its heading was
  never embedded → **chunking**.
- **rank** — does the chunk holding the answer rank within `top_k`? If it ranks too
  low → **retrieval**.

The chain stops at the first failing stage; every later stage is reported as *not
reached*, never as a false pass. The answer is supplied by you — nothing is inferred
from the model's own scores — so the verdict is grounded, not another guess.

## Install

```bash
pip install ragdoc
```

Requires Python 3.12. The first run downloads a small local embedding model
(`all-MiniLM-L6-v2`); every run after that is fully offline.

## Quick start

```bash
ragdoc diagnose paper.pdf "What was the sample size?" "384 participants"
```

```text
  extraction  ✓
  chunking    ✗  section heading not embedded in chunk text
  retrieval   ✗  target chunk ranked 23 of 400; precision@5 = 0.00 (max 0.20); RR = 0.04
  generation  — not reached

VERDICT: chunking — the answer's chunk sits under a heading that is not embedded
FIX: prepend section path to chunk text before embedding

Run configuration:
  loader: pymupdf
  chunk size: 500
  overlap: 50
  embedding model: all-MiniLM-L6-v2
  top_k: 5
  prepend_section_context: False
```

The answer can also be read from a file (handy for long or multi-line answers):

```bash
ragdoc diagnose paper.pdf "What was the sample size?" --answer-file answer.txt
```

Add `--verbose` for the full evidence behind each check, `--show-retrieved` to see the
retrieved chunks with the target marked, or `--ascii` for terminals that can't render
Unicode.

## Commands

### `diagnose` — one question, one verdict
```bash
ragdoc diagnose paper.pdf "What was the sample size?" "384 participants"
```

### `compare` — how different loaders extract the same answer
Silent extraction failures are often a loader problem. `compare` runs every PDF loader
against the same answer so you can see which one actually preserves it.
```bash
ragdoc compare paper.pdf "384 participants"
```

### `batch` — a failure distribution across a question set
One verdict tells you about one question; a *distribution* tells you where a pipeline
systematically breaks. Point `batch` at a JSON question set and it reports how the
failures split across the stages.
```bash
ragdoc batch paper.pdf questions.json --json results.json
```

The question set is a JSON array:
```json
[
  { "question": "What was the sample size?", "expected_answer": "384 participants" },
  { "question": "Which model performed best?", "expected_answer": "the gradient-boosted ensemble" }
]
```

### `sweep` — find the best configuration
Chunk size is a balance: too small splits answers apart (chunking failures), too big
dilutes them (retrieval failures). `sweep` runs the whole question set across a grid of
configurations and names the one that survives the most answers.
```bash
ragdoc sweep paper.pdf questions.json --chunk-size 500,800,1200 --overlap 0,50
```

It prints a table — one row per configuration, with the varied knobs, the no-failure
count, and the retrieval metrics — and highlights the winner. Use `--objective` to rank
by `no-failure` (default), `hit-rate`, or `mrr`.

## Configuration

Every knob has a sensible default and can be overridden on `diagnose` and `batch` (and
swept as a comma-separated list on `sweep`):

| Flag | Meaning | Default |
|---|---|---|
| `--chunk-size` | characters per chunk | `500` |
| `--overlap` | characters shared between adjacent chunks | `50` |
| `--top-k` | retrieval cutoff the ranking is judged against | `10` |
| `--embedding-model` | sentence-transformers model name | `all-MiniLM-L6-v2` |
| `--prepend-section-context` / `--no-prepend-section-context` | prepend the section path to chunk text before embedding | off |
| `--loader` | PDF extraction backend (`pymupdf`, `pdfplumber`, `pypdf`) | `pymupdf` |

## Known limitations

- **The answer must appear in the document text.** Matching is verbatim or normalized
  (whitespace, case, and Unicode compatibility forms such as ligatures are forgiven) —
  it does **not** handle paraphrase or reworded answers.
- **Scanned / image-only pages are not read.** Pages with no extractable text are
  flagged in an advisory, but `ragdoc` does no OCR.
- **The generation stage is not evaluated** — it is always reported as *not reached*.
  `ragdoc` diagnoses retrieval-and-earlier, where the failures it targets live.

## License

[MIT](LICENSE) © 2026 Ahmed Imran
