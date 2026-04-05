"""Verify that code examples in documentation execute and produce expected results."""

import re
from typing import Any

import pytest
from inline_snapshot import snapshot


def _extract_python_blocks(markdown: str) -> list[tuple[str, int]]:
    """Extract fenced Python code blocks from a Markdown string.

    Returns a list of (block_source, line_offset) tuples where line_offset is the
    number of lines preceding the block in the original file.
    """
    results = []
    for m in re.finditer(r"```python[^\n]*\n(.*?)```", markdown, re.DOTALL):
        line_offset = markdown[: m.start(1)].count("\n")
        results.append((m.group(1), line_offset))
    return results


def _exec_block(block: str, name: str, line_offset: int = 0) -> dict[str, Any]:
    """Execute a code block and return its namespace."""
    ns: dict[str, Any] = {}
    # Prepend empty lines so traceback line numbers match the source file.
    padded = "\n" * line_offset + block
    exec(compile(padded, name, "exec"), ns)
    return ns


QUICKSTART_EXPECTED_OUTPUT = snapshot("""\
Score: 1.42
  clever      sly
  sneaky      -
  fox         fox
  leaped      jumped
  -           across
""")


def test_quickstart_example(repo_root, capsys):
    blocks = _extract_python_blocks((repo_root / "README.md").read_text())
    block, start_line = blocks[0]
    _exec_block(block, "README.md", start_line)
    assert capsys.readouterr().out == QUICKSTART_EXPECTED_OUTPUT


USAGE_GLOBAL_ALIGNMENT_EXPECTED_OUTPUT = snapshot("""\
Episode I - The Phantom Menace       ->  --
Episode II - Attack of the Clones    ->  Attack of the Clones
Episode III - Revenge of the Sith    ->  --
Episode IV - A New Hope              ->  A New Hope
Episode V - The Empire Strikes Back  ->  The Empire Strikes Back
Episode VI - Return of the Jedi      ->  Return of the Jedi
""")


def test_usage_global_alignment(repo_root, capsys):
    fp = repo_root / "docs" / "USAGE.md"
    blocks = _extract_python_blocks(fp.read_text())

    block, start_line = blocks[0]
    _exec_block(block, fp, start_line)
    assert capsys.readouterr().out == USAGE_GLOBAL_ALIGNMENT_EXPECTED_OUTPUT


USAGE_SEMANTIC_DIFF_EXPECTED_OUTPUT = snapshot("""\
--- Semantic Document Diff ---

[ MATCH ] (sim: 0.86)
  - The company was founded in 2012 by two friends.
  + Two university buddies established the corp in 2012.

[ MATCH ] (sim: 0.81)
  - They started working in a small garage.
  + Their origins trace back to a tiny residential garage.

[ DELETED ]
  - Initially, they struggled to find investors.

[ MATCH ] (sim: 0.70)
  - However, their first product was a big hit.
  + After several months, they launched a successful app.

[ INSERTED ]
  + They recently expanded to the European market.

[ MATCH ] (sim: 0.91)
  - Today, they employ over 500 people.
  + Currently, the workforce consists of 500+ employees.

""")


@pytest.mark.slow
def test_usage_semantic_diff(repo_root, capsys):
    pytest.importorskip("fastembed")

    fp = repo_root / "docs" / "USAGE.md"
    blocks = _extract_python_blocks(fp.read_text())

    block, start_line = blocks[1]
    _exec_block(block, fp, start_line)
    assert capsys.readouterr().out == USAGE_SEMANTIC_DIFF_EXPECTED_OUTPUT
