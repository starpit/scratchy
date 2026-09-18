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
    # util/foldManager/ was UNSCOPED in the first sc1 and three agents correctly reported BLOCKED
    # rather than invent FoldManager: CoordinateType's `coordinates_` is
    # std::map<PrimaryDimTypes, FoldManager<Dtype>> and ~13 of its ~20 methods route through it, so
    # via it the whole eight-kind node union was unportable without breaking Rule 1 or Rule 2.
    "foldInfrastructure.h": "schedule/fold.rs",
    "mapWithFMHelper.h": "schedule/fold_helper.rs",
    "wkDivisionParams.h": "schedule/wk_division.rs",
    "dsc2.h": "schedule/dsc2.rs",
    "designSpaceConfig.h": "schedule/dsc.rs",
    "dims.h": "schedule/dims.rs",
    "ddc_metadata.h": "schedule/metadata.rs",
    "ddc.h": "schedule/ddc.rs",
}
BUDGETS = {"max_syms": 8, "max_loc": 320, "max_types": 5, "min_fields": 20}
MIN_BODY = 4          # a 3-line class is a tag or a typedef, not a unit of work
FIELD = re.compile(r"\b([a-zA-Z][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*(?:=[^;]*)?;\s*$")


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
            # ⛔ AND A PARAMETER CONTINUATION IS NOT A FIELD. Dropping the trailing-underscore
            # requirement let `const bool isPadFront = true);` — the tail of a method signature
            # wrapped across lines — contribute `isPadFront`, and `const` besides. A parameter list's
            # continuation ends in `);` or `,`; a member declaration does not.
            KW = {"const", "static", "inline", "return", "struct", "class", "using", "typedef",
                  "public", "private", "protected", "virtual", "explicit", "friend", "template"}
            flds = []
            for bl in body.splitlines():
                if re.search(r"return|\(|\.|->|&|\[\"", bl) or bl.lstrip().startswith("//"):
                    continue
                if re.search(r"\)\s*;?\s*$|,\s*$", bl):
                    continue
                fm = FIELD.search(bl)
                if fm and fm.group(1) not in KW:
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

    # ⛔⛔ AN ORDER-FORCING EDGE IS **STRUCTURAL**, NOT ANY MENTION. Folding every `Ret Class::method`
    # body into the edges made the graph nearly complete: Tarjan then found ONE strongly-connected
    # component of NINETEEN classes — the whole node hierarchy plus DesignSpaceConfig, Ddc and
    # Metadata — which is true of the C++ and useless as a batch.
    #
    # ⭐ What actually constrains a Rust port is narrower: types in the SAME MODULE may reference each
    # other freely, so a method mentioning a type forces nothing. Only a BASE CLASS and a FIELD's TYPE
    # do — a base because the derived type cannot exist without it, a field because the struct cannot
    # be written without the type it holds. Method-body mentions still matter for scope (they are why
    # foldManager had to be scoped at all), but they are not an ordering signal.
    for c in cs.values():
        decl_lines = []
        for bl in c["body"].splitlines():
            if re.search(r"return|\(|\.|->|\[\"", bl) or bl.lstrip().startswith("//"):
                continue
            if re.search(r"\)\s*;?\s*$|,\s*$", bl):
                continue
            if re.search(r"\b[a-zA-Z][A-Za-z0-9_]*\s*(?:\[[^\]]*\])?\s*(?:=[^;]*)?;\s*$", bl):
                decl_lines.append(bl)
        c["code"] = "\n".join(decl_lines)
        c["deps"] = sorted(
            n for n in names
            if n != c["name"] and re.search(rf"\b{re.escape(n)}\b", c["code"])
        )

    # ⛔ AN INHERITANCE EDGE IS NEVER BROKEN. The base is named on the `class X : public Base…` line,
    # and a derived class cannot land before it.
    for c in cs.values():
        decl = c["body"].splitlines()[0]
        colon = decl.find(":")
        c["bases"] = sorted(n for n in names
                            if n != c["name"] and colon >= 0
                            and re.search(rf"\b{re.escape(n)}\b", decl[colon:]))
        for b in c["bases"]:
            if b not in c["deps"]:
                c["deps"].append(b)

    # ⛔⛔ A MUTUALLY RECURSIVE CLUSTER IS **ONE BATCH**, NOT AN ORDERING PROBLEM. Breaking its cycles
    # to pick a winner is the wrong answer twice over: whichever member lands first is written against
    # types that do not exist yet, and the tie-break has no honest signal. Body size was tried and is
    # actively wrong — ScheduleTree (32 lines) CONTAINS ScheduleNode (81), so "the container is bigger"
    # put the tree first. Three review agents said it plainly: e007/e013/e015/e017 are one cluster the
    # plan "wrongly split across four waves". So: condense each strongly-connected component to a
    # single scheduling node, give it one layer and one batch, and let one agent write the members
    # together — which is what mutual recursion actually requires.
    index, stack, on, low, num, sccs = [0], [], set(), {}, {}, []

    def strong(v):                                    # Tarjan, iterative-safe depth here (≤ 44 nodes)
        low[v] = num[v] = index[0]; index[0] += 1
        stack.append(v); on.add(v)
        for w in cs[v]["deps"]:
            if w not in num:
                strong(w); low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], num[w])
        if low[v] == num[v]:
            comp = []
            while True:
                w = stack.pop(); on.discard(w); comp.append(w)
                if w == v:
                    break
            sccs.append(sorted(comp))

    for n in cs:
        if n not in num:
            strong(n)

    comp_of = {n: i for i, comp in enumerate(sccs) for n in comp}
    comp_deps = {i: set() for i in range(len(sccs))}
    for n, c in cs.items():
        for d_ in c["deps"]:
            if comp_of[d_] != comp_of[n]:
                comp_deps[comp_of[n]].add(comp_of[d_])

    cdepth: dict[int, int] = {}

    def cd(i):
        if i not in cdepth:
            cdepth[i] = 0                             # set before recursing: the condensation is a DAG
            cdepth[i] = max((cd(j) + 1 for j in comp_deps[i]), default=0)
        return cdepth[i]

    for i in range(len(sccs)):
        cd(i)
    for n in cs:
        cs[n]["scc"] = comp_of[n]

    depth = {n: cdepth[comp_of[n]] for n in cs}
    multi = [c for c in sccs if len(c) > 1]
    if multi:
        print("  mutually recursive clusters, each ONE batch:")
        for c in multi:
            print(f"    L{cdepth[comp_of[c[0]]]}: " + ", ".join(c))

    order = sorted(cs.values(), key=lambda c: (depth[c["name"]], c["header"], c["first"]))
    rows = ["\t".join(["entry", "level", "loc", "authority", "extract_lines", "rust_home",
                       "callees", "note"])]
    for i, c in enumerate(order, 1):
        e = f"e{i:03d}_{c['name']}"
        c["entry"] = e
        c["home"] = HOME[c["header"]]
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

    # ⭐ AND A REMAINDER, or a re-run re-ports what already landed. The driver's stage() sees a .done
    # marker beside a non-empty schedule and calls the marker stale, so pointing it at port.json would
    # hand 21 finished units back to agents. Drop any unit whose `/// Replaces:` anchor is already in
    # its home.
    # ⛔ AN ANCHOR IS NOT A PORT — it is only the signal crustify can consume. The gate stays: zero
    # todo!, zero traits, a real caller, and nodes minted on a real program.
    landed = set()
    for home in {c["home"] for c in order}:
        f = ROOT / "crates/compiler/deeptools/src" / home
        if f.exists():
            # ⛔ MATCH ON THE CLASS NAME, NOT THE eNNN NUMBER. Scoping foldManager inserted 12 units
            # and renumbered every later one, so wave 1's anchors stopped identifying their units —
            # only 13 of the 21 landed types still matched, and those by coincidence. The number is an
            # ordering artefact; the class name is the identity.
            landed |= {a.split("_", 1)[1] for a in
                       re.findall(r"/// Replaces: (e\d{3}_[A-Za-z0-9_]+)", f.read_text())}
    rem_waves = []
    for w in doc["waves"]:
        bs = []
        for b in w["batches"]:
            keep = [i for i in b["items"] if i["name"].split("_", 1)[1] not in landed]   # not `items`: that shadowed
            if keep:                                                     # the outer list and made the
                bs.append(dict(b, items=keep))                           # summary report 4 anchors
        if bs:
            rem_waves.append({"unit_count": sum(len(b["items"]) for b in bs), "batches": bs})
    ritems = [i for w in rem_waves for b in w["batches"] for i in b["items"]]
    rdoc = dict(doc, waves=rem_waves, summary={
        "unit_count": len(ritems),
        "layer_count": len({i["layer"] for i in ritems}),
        "batch_count": sum(len(w["batches"]) for w in rem_waves),
        "file_count": len({b["source_file"] for w in rem_waves for b in w["batches"]}),
    })
    (out / "port-remainder.json").write_text(json.dumps(rdoc, indent=1) + "\n")
    print(f"  remainder: {len(ritems)} units left ({len(landed)} anchored), "
          f"{rdoc['summary']['batch_count']} batches, {rdoc['summary']['layer_count']} layers")
    s = doc["summary"]
    print(f"  UNITS.tsv: {len(order)} classes")
    print(f"  sc1-dsc-data-model: {s['unit_count']} units, {len(waves)} waves, "
          f"{s['batch_count']} batches, {s['layer_count']} layers, "
          f"{sum(len(i['field_anchors']) for i in items)} field anchors")
    for layer in sorted(by_layer):
        print(f"    L{layer}: " + ", ".join(c["name"] for c in by_layer[layer]))


main()
