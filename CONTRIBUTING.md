# Contributing to pynw

This project is primarily a personal playground for experimenting with
different technologies and methodologies (e.g. Rust, PyO3, agentic
engineering). That said, if you find it useful and want to contribute,
contributions are welcome!

## Reporting Issues

If you find a bug or have a feature request, please open an
[issue](https://github.com/chrisdeutsch/pynw/issues).

## Submitting Changes

For larger features or significant changes, please open an
[issue](https://github.com/chrisdeutsch/pynw/issues) first to discuss whether
the change aligns with the project's goals.

1. Fork the repository and create a branch from `main`.
2. Make your changes.
3. Ensure all checks pass (see below).
4. Open a pull request with a clear description of what you changed and why.

## Development Setup

This project uses [pixi](https://pixi.sh) for environment management. To get
started:

```bash
pixi install
pixi run build
```

## Running Checks

Before submitting a pull request, make sure the following pass:

```bash
pixi run test             # deterministic tests
pixi run test -- -m hypothesis  # property-based tests
pixi run lint             # formatting & linting (ruff, cargo fmt, prettier, markdownlint, taplo, actionlint)
pixi run check            # static analysis (clippy, mypy)
```

[Lefthook](https://github.com/evilmartians/lefthook) is configured to run
formatting and linting checks automatically on commit and push.

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
