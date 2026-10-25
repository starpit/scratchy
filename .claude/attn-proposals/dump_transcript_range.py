#!/usr/bin/env python3
"""Dump a transcript line-range around a given line: types, tool uses, text, thinking."""
import json
import sys

path = sys.argv[1]
start, end = int(sys.argv[2]), int(sys.argv[3])

with open(path) as f:
    for i, line in enumerate(f, 1):
        if i < start or i > end:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        t = o.get("type")
        if t == "assistant":
            for c in o.get("message", {}).get("content", []):
                ct = c.get("type")
                if ct == "tool_use":
                    inp = c.get("input", {})
                    print(f"[{i}] TOOL_USE {c.get('name')} {inp.get('file_path','') or inp.get('description','')}")
                    for k in ("old_string", "new_string"):
                        v = inp.get(k)
                        if v:
                            print(f"      {k}: {str(v)[:200]}")
                elif ct == "text":
                    print(f"[{i}] TEXT: {str(c.get('text',''))[:600]}")
                elif ct == "thinking":
                    print(f"[{i}] THINKING: {str(c.get('thinking',''))[:600]}")
        elif t == "queue-operation":
            c = str(o.get("content", ""))[:1500]
            print(f"[{i}] QUEUE: {c}")
        elif t == "user":
            content = o.get("message", {}).get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "tool_result":
                        print(f"[{i}] TOOL_RESULT: {str(c.get('content'))[:300]}")
