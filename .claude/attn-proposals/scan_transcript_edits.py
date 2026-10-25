#!/usr/bin/env python3
"""Scan a transcript for Edit/Write/Bash tool_use events touching placement.rs or its derive line."""
import json
import sys

path = sys.argv[1]
show_all = "-a" in sys.argv

with open(path) as f:
    for i, line in enumerate(f, 1):
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("type") != "assistant":
            continue
        for c in o.get("message", {}).get("content", []):
            if c.get("type") != "tool_use":
                continue
            name = c.get("name")
            inp = c.get("input", {})
            if name in ("Edit", "Write", "NotebookEdit"):
                fp = str(inp.get("file_path", ""))
                if "placement.rs" not in fp:
                    continue
                blob = str(inp.get("old_string", "")) + str(inp.get("new_string", "")) + str(inp.get("content", ""))
                hit = "PartialOrd" in blob or "BindCover" in blob or show_all
                if hit:
                    print(f"line {i}: {name} {fp}")
                    for k in ("old_string", "new_string", "content"):
                        v = inp.get(k)
                        if v:
                            s = str(v)
                            print(f"  {k} ({len(s)} ch): {s[:220]}")
            elif name == "Bash":
                cmd = str(inp.get("command", ""))
                if "placement.rs" in cmd and ("rustfmt" in cmd or "git" in cmd):
                    print(f"line {i}: Bash {cmd[:300]}")
