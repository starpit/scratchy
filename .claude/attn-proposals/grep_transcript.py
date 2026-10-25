#!/usr/bin/env python3
"""Print transcript lines mentioning a term, with brief context of what each event is."""
import json
import sys

path = sys.argv[1]
term = sys.argv[2]
maxout = int(sys.argv[3]) if len(sys.argv) > 3 else 40

n = 0
with open(path) as f:
    for i, line in enumerate(f, 1):
        if term not in line:
            continue
        n += 1
        if n > maxout:
            print(f"... (more, {n-1} shown)")
            break
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
                    blob = json.dumps(inp)
                    if term in blob:
                        idx = blob.find(term)
                        print(f"[{i}] TOOL {c.get('name')}: ...{blob[max(0,idx-150):idx+250]}...")
                elif ct in ("text", "thinking"):
                    s = str(c.get(ct, ""))
                    if term in s:
                        idx = s.find(term)
                        print(f"[{i}] {ct.upper()}: ...{s[max(0,idx-200):idx+400]}...")
        elif t == "queue-operation":
            c = str(o.get("content", ""))
            idx = c.find(term)
            print(f"[{i}] QUEUE: ...{c[max(0,idx-200):idx+400]}...")
        elif t == "user":
            content = o.get("message", {}).get("content")
            s = str(content)
            if term in s:
                idx = s.find(term)
                print(f"[{i}] USER: ...{s[max(0,idx-150):idx+300]}...")
