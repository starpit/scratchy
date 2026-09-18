#!/usr/bin/env python3
"""Emit the scheduler re-port's units and wave schedules from the vendored C++ headers.

⛔ WHY THIS IS TYPE-LED. The deleted port scheduled FUNCTIONS one at a time. Each was ported
against facts its type did not carry, so the fact became a trait method and the trait became a
carrier file: 20,494 lines of `schedule/stages/*.rs` with no C++ counterpart, holding 130
`todo!()`s. A unit here is a CLASS — its fields AND its methods, together — so a method never
has to ask another type for a fact its own type should hold.

Levels are the longest path over MEASURED edges: each class body is scanned for the names of
other in-scope classes. Nothing is hand-assigned. A wave is a barrier, so a class always lands
strictly after every class it names.
"""
import hashlib
import json
import pathlib
import re

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent.parent.parent
CPP = ROOT / "crustify-scheduler" / "cpp"
UNITS = ROOT / "crustify-scheduler" / "UNITS.tsv"

# One Rust module per C++ header: the header's classes land together, as they are declared
# together. No module exists for a header with no scheduled class.
HOME = {
    "dsc2.h": "schedule/dsc2.rs",
    "designSpaceConfig.h": "schedule/dsc.rs",
    "dims.h": "schedule/dims.rs",
    "ddc_metadata.h": "schedule/metadata.rs",
    "ddc.h": "schedule/ddc.rs",
}
BUDGETS = {"max_syms": 8, "max_loc": 320, "max_types": 5, "min_fields": 20}
MIN_BODY = 4          # a 3-line class is a tag or a typedef, not a unit of work
FIELD = re.compile(r"\b([a-zA-Z][a-zA-Z0-9]*_)\s*(?:\[[^\]]*\])?\s*(?:=[^;]*)?;")


def classes():
    """Every class in the vendored headers with a real body, and its declared fields."""
    out = {}
    for h in sorted(CPP.glob("*.h")):
        if h.name not in HOME:
            continue
        lines = h.read_text(errors="replace").splitlines()
        for i, ln in enumerate(lines):
            m = re.match(r"^(class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(:|\{|$)", ln)
            if not m:
                continue
            end = next((j for j in range(i, len(lines)) if lines[j].startswith("};")), None)
            if end is None or end - i + 1 < MIN_BODY:
                continue
            # `template<> struct std::hash<PrimaryDimAndKind>` matches the class regex with the
            # name `std`. A namespace is not a unit; neither is any name that is not a bare
            # identifier the port can own.
            if m.group(2) in ("std",):
                continue
            body = "\n".join(lines[i:end + 1])
            # ⛔ DECLARATIONS ONLY. Matching any `name_` before `;`/`=`/`[` also catches member
            # ACCESSES in the class's own inline methods — `&N_.i_`, `primaryDsInfo_.at(x).f_` —
            # which belong to NESTED types. That inflated DesignSpaceConfig 36 -> 49 and the
            # campaign total 171 -> 197, and was quoted as a real gap for a whole session.
            flds = []
            for bl in body.splitlines():
                if re.search(r"return|\(|\.|->|&|\[\"", bl) or bl.lstrip().startswith("//"):
                    continue
                fm = FIELD.search(bl)
                if fm:
                    flds.append(fm.group(1))
            out[m.group(2)] = {
                "name": m.group(2), "header": h.name, "kind": m.group(1),
                "first": i + 1, "last": end + 1, "loc": end - i + 1,
                "fields": sorted(set(flds)), "body": body,
            }
    return out


def main():
    cs = classes()
    names = set(cs)
    # MEASURED edges: a class depends on another in-scope class its body names. Self-reference and
    # a base class named on the `class X : public Y` line both count — a derived class cannot land
    # before its base.
    # ⛔ STRIP COMMENTS BEFORE MEASURING AN EDGE. A class name mentioned in a `//` or `/* */`
    # comment is not a dependency, and counting it fabricated edges that turned 33 classes into 14
    # near-serial waves — a class per wave, which is the slowest possible schedule and wrong about
    # the graph besides.
    # ⛔⛔ A CLASS'S DEPENDENCIES ARE MOSTLY IN ITS .cpp, NOT ITS HEADER. Measuring edges from the
    # header alone put DesignSpaceConfig at L4 — five layers ahead of the AllocateNode that
    # getBufferCapacityForNode takes — because that method is DECLARED in designSpaceConfig.h and
    # DEFINED in dsc2.cpp, so the type never appears in the class body at all. Scan every
    # `Ret Class::method(..)` body in the authority tree and fold what it names into the class's
    # edges. This is the one thing a header scan cannot get right and the CodeQL oracle would.
    AUTH = pathlib.Path("/Users/nickm/git/deeptools-src")
    impl_text = {}
    for rel in ("dsc/dsc2.cpp", "dsc/designSpaceConfig.cpp", "dsc/dims.cpp",
                "ddc/ddcv1.cpp", "ddc/ddc_fold.cpp", "ddc/ddc_metadata.cpp"):
        p = AUTH / rel
        if p.exists():
            impl_text[rel] = p.read_text(errors="replace")

    def method_bodies(cls):
        """Every `… Cls::name(…) { … }` body across the authority .cpp files, concatenated."""
        acc = []
        for txt in impl_text.values():
            for m in re.finditer(rf"^[A-Za-z_][\w:<>,*& ]*\b{re.escape(cls)}::", txt, re.M):
                start = m.start()
                brace = txt.find("{", start)
                if brace < 0:
                    continue
                depth_, j = 0, brace
                while j < len(txt):
                    if txt[j] == "{":
                        depth_ += 1
                    elif txt[j] == "}":
                        depth_ -= 1
                        if depth_ == 0:
                            break
                    j += 1
                acc.append(txt[start:j + 1])
        return "\n".join(acc)

    for c in cs.values():
        code = re.sub(r"/\*.*?\*/", " ", c["body"] + "\n" + method_bodies(c["name"]), flags=re.S)
        code = "\n".join(re.sub(r"//.*", "", bl) for bl in code.splitlines())
        c["code"] = code
        c["impl_loc"] = len(method_bodies(c["name"]).splitlines())
        c["deps"] = sorted(
            n for n in names
            if n != c["name"] and re.search(rf"\b{re.escape(n)}\b", code)
        )

    # Longest-path depth over those edges. Mutual references are common and real here — a node names
    # the config and the config's methods take the node — so a cycle is broken by BODY SIZE: the
    # bigger class is the container and lands later.
    #
    # ⛔ The first version broke ties on the declaration line `first`, which compares line numbers
    # across DIFFERENT headers and is therefore meaningless. It put DesignSpaceConfig
    # (designSpaceConfig.h:51, 650 lines, holds the schedule tree) at L4, five layers BEFORE the
    # AllocateNode (dsc2.h:974, 84 lines) that its own getBufferCapacityForNode takes — handing an
    # agent a class whose methods need a type that does not exist yet, in a campaign whose Rule 2
    # forbids inventing one. Size is not a guess: a container is larger than the thing it contains.
    def heavier(a, b):
        return (cs[a]["loc"], a) > (cs[b]["loc"], b)

    # ⛔ BREAK THE CYCLES FIRST, THEN MEASURE DEPTH. Deciding inside the recursion does not work:
    # the `x in seen` guard fires before any size test, so which member of a cycle lands first
    # depends on which node the walk happened to enter from. That is how DesignSpaceConfig kept
    # coming out at L4, five layers ahead of the AllocateNode its own methods take, no matter what
    # tie-break was written. For each mutual pair keep only heavy -> light, so the container is the
    # consumer and lands strictly later; the result is a DAG and depth is then just longest path.
    # ⛔ AN INHERITANCE EDGE IS NEVER BROKEN. The base is named on the `class X : public Base…` line,
    # and a derived class cannot land before it. Size is the wrong signal here — a base is usually
    # LARGER than what derives from it, so the heuristic put BlockNode (36 lines) at L1 and the
    # ScheduleNode (81 lines) it derives from at L3. The previous campaign hit exactly this: BlockNode
    # scheduled in the same wave as its base, which hands an agent a type whose parent does not exist.
    for c in cs.values():
        decl = c["body"].splitlines()[0]
        colon = decl.find(":")
        c["bases"] = sorted(n for n in names
                            if n != c["name"] and colon >= 0
                            and re.search(rf"\b{re.escape(n)}\b", decl[colon:]))
        for b in c["bases"]:
            if b not in c["deps"]:
                c["deps"].append(b)

    for a in cs:
        for b in list(cs[a]["deps"]):
            if b in cs[a]["bases"]:
                continue                                   # inheritance: keep, always
            if a in cs[b]["deps"] and a not in cs[b]["bases"] and not heavier(a, b):
                cs[a]["deps"].remove(b)

    depth: dict[str, int] = {}

    def d(n, seen=()):
        if n in depth:
            return depth[n]
        if n in seen:                      # a cycle of three or more; rare here, and harmless
            return 0
        depth[n] = max((d(x, seen + (n,)) + 1 for x in cs[n]["deps"]), default=0)
        return depth[n]

    for n in cs:
        d(n)

    order = sorted(cs.values(), key=lambda c: (depth[c["name"]], c["header"], c["first"]))
    rows = ["\t".join(["entry", "level", "loc", "authority", "extract_lines", "rust_home",
                       "callees", "note"])]
    for i, c in enumerate(order, 1):
        e = f"e{i:03d}_{c['name']}"
        c["entry"] = e
        note = (f"{c['kind']} {c['name']}, {c['loc']} lines, {len(c['fields'])} declared fields. "
                f"PORT ITS FIELDS AND ITS METHODS TOGETHER — every method defined in the header and "
                f"every method defined in the matching .cpp. A fact a method needs is a FIELD on "
                f"this type, ported now; it is never a trait method and never a parameter someone "
                f"else supplies.")
        rows.append("\t".join([
            e, str(depth[c["name"]]), str(c["loc"]),
            f"dsc/{c['header']}" if c["header"] in ("dsc2.h", "designSpaceConfig.h", "dims.h")
            else f"ddc/{c['header']}",
            f"{c['first']}-{c['last']}", HOME[c["header"]],
            ",".join(x for x in c["deps"] if depth[x] < depth[c["name"]]), note,
        ]))
    UNITS.write_text("\n".join(rows) + "\n")

    by_layer: dict[int, list] = {}
    for c in order:
        by_layer.setdefault(depth[c["name"]], []).append(c)

    waves = []
    for layer in sorted(by_layer):
        grouped: dict[str, list] = {}
        for c in by_layer[layer]:
            grouped.setdefault(c["header"], []).append(c)
        batches = []
        for hdr, gs in grouped.items():
            cur, nf = [], 0
            for c in gs:
                if cur and (len(cur) >= BUDGETS["max_types"] or nf >= BUDGETS["min_fields"]):
                    batches.append((hdr, cur)); cur, nf = [], 0
                cur.append(c); nf += len(c["fields"])
            if cur:
                batches.append((hdr, cur))
        waves.append({
            "unit_count": len(by_layer[layer]),
            "batches": [{
                "kind": "type", "source_file": f"crustify-scheduler/cpp/{hdr}",
                "items": [{
                    "name": c["entry"],
                    "defined_in": f"crustify-scheduler/cpp/{c['header']}",
                    "kind": "type", "source_kind": "struct",   # crustify's taxonomy is C's: it
                    # rejects a whole batch on `unsupported type kind(s) ['class']`, and a C++ class
                    # differs from a struct only in default member access.
                    "layer": layer, "loc": c["loc"],
                    "deps": {"types": [{
                        "name": cs[x]["entry"],
                        "defined_in": f"crustify-scheduler/cpp/{cs[x]['header']}",
                        "scope": "port",
                    } for x in c["deps"] if depth[x] < layer], "symbols": []},
                    "fallback": [], "back_fill": [], "generates": [],
                    "field_anchors": c["fields"],
                } for c in b],
            } for hdr, b in batches],
        })

    items = [i for w in waves for b in w["batches"] for i in b["items"]]
    doc = {
        "schema_version": 3,
        "oracle_config": {
            # crustify validates this against the file on disk: 64 lowercase hex,
            # or the schedule is rejected as having no oracle binding at all.
            "path": "crustify-scheduler/scope-config.json",
            "sha256": hashlib.sha256(
                (ROOT / "crustify-scheduler" / "scope-config.json").read_bytes()).hexdigest(),
        },
        "api_headers_only": False, "budgets": dict(BUDGETS),
        "summary": {
            "unit_count": len(items),
            "layer_count": len({i["layer"] for i in items}),
            "batch_count": sum(len(w["batches"]) for w in waves),
            "file_count": len({b["source_file"] for w in waves for b in w["batches"]}),
        },
        "waves": waves,
    }
    ids = [(i["name"], i["defined_in"]) for i in items]
    assert len(set(ids)) == len(ids), "duplicate (name, defined_in)"
    assert len(items) == doc["summary"]["unit_count"]
    seen: set[str] = set()
    for w in waves:
        for b in w["batches"]:
            for i in b["items"]:
                for dep in (x["name"] for x in i["deps"]["types"]):
                    assert dep in seen, f"{i['name']} depends on {dep} in the same or a later wave"
        seen |= {i["name"] for b in w["batches"] for i in b["items"]}

    out = BASE / "sc1-dsc-data-model"
    out.mkdir(exist_ok=True)
    for obj in ("port", "review"):
        (out / f"{obj}.json").write_text(json.dumps(doc, indent=1) + "\n")
    s = doc["summary"]
    print(f"  UNITS.tsv: {len(order)} classes")
    print(f"  sc1-dsc-data-model: {s['unit_count']} units, {len(waves)} waves, "
          f"{s['batch_count']} batches, {s['layer_count']} layers, "
          f"{sum(len(i['field_anchors']) for i in items)} field anchors")
    for layer in sorted(by_layer):
        print(f"    L{layer}: " + ", ".join(c["name"] for c in by_layer[layer]))


main()
