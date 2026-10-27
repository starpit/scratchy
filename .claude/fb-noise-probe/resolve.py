#!/usr/bin/env python3
"""Apply per-hunk side choices to a git-conflicted file (front-b rebase).

Usage: resolve.py <file> <choices>
  choices: comma-separated per-hunk 'H' (HEAD/Front A) or 'T' (theirs/mine).
Prints each hunk's first line before applying, as the read-back check.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1])
choices = sys.argv[2].split(",")
text = path.read_text()
lines = text.splitlines(keepends=True)

out = []
hunk = None
side = None  # None outside; "H" or "T" inside
idx = -1
for line in lines:
    if line.startswith("<<<<<<< "):
        idx += 1
        assert idx < len(choices), f"hunk {idx + 1} has no choice"
        hunk = []
        side = "H"
        continue
    if line.startswith(">>>>>>> "):
        h = choices[idx]
        assert h in ("H", "T"), f"bad choice {h!r} for hunk {idx + 1}"
        kept = [l for s, l in hunk if s == h]
        out.extend(kept)
        hunk = None
        side = None
        continue
    if line.rstrip("\n") == "=======":
        side = "T"
        continue
    if hunk is not None:
        hunk.append((side, line))
    else:
        out.append(line)

assert hunk is None, "unterminated hunk"
assert idx == len(choices) - 1, f"{idx + 1} hunks found, {len(choices)} choices given"
path.write_text("".join(out))
print(f"{path}: resolved {idx + 1} hunk(s) with {''.join(choices)}")
