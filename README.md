# ragdoc

A headless diagnostic for PDF RAG pipelines. `ragdoc` runs the forward pipeline
(load → chunk → index → retrieve) and a backward set of checks that decide which
stage lost a known answer, returning a verdict with evidence.

This is the v0 engine. See `specs/001-v0-headless-attribution-engine/` for the
specification and plan, and `.ai/` for the project's durable design context.

## Requirements

Python 3.12 (the embedding stack used from T-005 does not reliably install on
3.14).

## Install

```
pip install -e .
```
