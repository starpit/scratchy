#!/usr/bin/env python3
"""Inspect transcript events: print type/tool/text for given 1-based line numbers."""
import json
import sys

path = sys.argv[1]
nums = [int(a) for a in sys.argv[2:]]

with open(path) as f:
    for i, line in enumerate(f, 1):
        if i not in nums:
            continue
        print(f"=== line {i} ===")
        try:
            o = json.loads(line)
        except Exception as e:
            print("PARSE ERR", e)
            continue
        t = o.get("type")
        if t == "assistant":
            for c in o.get("message", {}).get("content", []):
                if c.get("type") == "tool_use":
                    inp = c.get("input", {})
                    print("TOOL_USE", c.get("name"), inp.get("file_path", ""))
                    for k in ("old_string", "new_string", "command", "description"):
                        v = inp.get(k)
                        if v:
                            print(f"  {k}: {str(v)[:400]}")
                elif c.get("type") == "text":
                    print(f"TEXT: {str(c.get('text',''))[:400]}")
        elif t == "user":
            content = o.get("message", {}).get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "tool_result":
                        print("TOOL_RESULT:", str(c.get("content"))[:400])
                    elif isinstance(c, dict) and c.get("type") == "text":
                        print("USER TEXT:", str(c.get("text"))[:400])
            else:
                print("USER:", str(content)[:400])
        else:
            print(t, str(o)[:300])
