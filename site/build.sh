#!/usr/bin/env bash
# Assemble the GitHub Pages site. See build.py for what it actually does —
# markdown rendering is client-side (zero-md), so there is no mdBook step.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"

python3 "$here/build.py"

# Generated straight from the DSL sources, so the architectures page cannot
# drift from what the compiler actually reads.
python3 "$here/build-archs.py" "$root" "$here/_site/architectures.html"
