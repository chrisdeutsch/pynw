# pynw

General-purpose, domain-agnostic sequence alignment library. The API operates
on caller-supplied score matrices - domain-specific logic (tokenization,
scoring) belongs in the caller. Rust core (PyO3) + Python bindings.

## Architecture

- `src/` — Rust DP implementations + PyO3 bindings (`lib.rs` exposes to Python)
- `pynw/_native.pyi` — type stubs (keep in sync with Rust doc comments in `src/lib.rs`)
- `pynw/__init__.py` — public API re-exports; update `__all__` when adding functions
- `tests/` — pytest suite

Most code should be implemented in the Rust crate. The Python library only
defines the API and implements convenience functionality.

## Build & Test

Requires `pixi` (see `pixi.toml`).

- `pixi run build`: compile extension (`maturin develop --release`)
- `pixi run test`: run pytest (builds first)
- `pixi run test-hypothesis`: run hypothesis property tests only
- `pixi run lint`: run pre-commit hooks (ruff, cargo fmt, markdownlint, taplo)
- `pixi run check`: run pre-push hooks (clippy, mypy)
