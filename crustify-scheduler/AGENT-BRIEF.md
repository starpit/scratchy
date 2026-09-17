# crustify-scheduler — re-port IBM's scheduler and bridge 1, as the C++ is shaped

This is a **re-port**. The previous attempt turned ~30,000 lines of working C++ into **114,193 lines
of Rust that did not work**, and it was deleted whole on 2026-09-17 (tag
`scheduler-before-nuke-2026-09-17`). Read the two prohibitions below before anything else. They are
not style preferences; each one names the mechanism that produced that failure.

## ⛔⛔ RULE 1 — YOU CAN NEVER INTRODUCE A TRAIT

No `trait`. No generic type parameter standing in for data. No `impl SomeTrait for SomeType` to supply
a fact. No "provider", "carrier", "store", "reads", "sites" or "state" type whose job is to answer
questions on behalf of another type.

**Why:** the deleted port made the algorithms generic over eight traits — `Dsc2Store`, `Dsc2Reads`,
`Dsc2Tree`, `Dsc2Stages`, `Dsc2Sites`, `Dsc2State`, `DdcState`, `DscState`. Every C++ member access
`dsc.foo_` became a trait method, and something had to implement it, so **20,494 lines of
`schedule/stages/*.rs` came into existence with no C++ counterpart at all** — 18% of the crate,
holding **130 `todo!()`s**, 129 of them a bare `todo!()` with no citation because *there was no C++
line to cite*. Five days of work moved that count from 131 to 130.

⭐ In the reference there is ONE object and the scheduler calls methods on it. Do that. A fact a
method needs is a **field on the type**, ported now, in the same unit.

## ⛔⛔ RULE 2 — YOU CAN NEVER MAKE UP A TYPE

Every type name, field name and method name you write must be one of exactly two things:

1. declared in the C++ authority, or
2. already present in `crates/compiler/deeptools/src`.

If neither, you do not invent it — you stop and say so in your report.

**Why:** the last unit to fail did so three times in a row, and each time the agent reached for
`DscInputs`, `DscDims`, `dsc_names` and a `full_padding` field. **All four have zero hits tree-wide.**
It burned ~4.5 hours and committed nothing. Earlier in the same effort a port called
`with_placed`/`write_placed`, helpers that did not exist either.

⛔ "It compiles" does not clear this rule and neither does a plausible name. If you cannot find the
C++ declaration, cite what you searched and stop.

## ⛔ RULE 3 — A UNIT IS A CLASS WITH ITS FIELDS **AND** ITS METHODS, TOGETHER

Never one function at a time. Function-at-a-time scoping is precisely what built the carrier layer:
each function was ported against facts its type did not carry, the fact became a trait method, and the
trait became a file. The class declaration and its method bodies are one piece of work.

⭐ The header and the `.cpp` are listed side by side in the oracle inventory for this reason. If your
unit's methods read a field, that field is yours to port in the same unit.

## ⛔ RULE 4 — ONE OBJECT, NOT TWO `&mut` PARAMETERS

`run_v1(SuperDsc&)` and `L3DlOpsScheduler::run` mutate ONE object. The deleted port took
`run_v1(sdsc: &mut SuperDsc, sites: &mut P, ..)` — two independent `&mut` — so the borrow checker
forced `currDsc` to be a **clone**, and every write through it was silently dropped while still
compiling. That file's own header admitted it: *"those writes do NOT reach this clone"* and *"no caller
can close it"*. Whatever signature you choose, the object the reference mutates is ONE owned value.

## ⛔ RULE 5 — NOTHING IS WIRED UP LATER

No `todo!("wants X — see Y")`. No field left absent with a note explaining who should have filled it.
No stub whose message describes the fact instead of carrying it. If you cannot finish the unit, your
report says which C++ line defeated you and why — you do not leave a placeholder that reads like
progress.

## The authority

`/Users/nickm/git/deeptools-src/<file>:<line>`, revision `a0d29abbed`. Cite `path:line` in every doc
comment. ⛔ `/Users/nickm/git/deeptools` is a DIFFERENT revision — do not use it.

⛔ **Your own tree's doc comments are not evidence.** Ten defects in the deleted port were found by
reading IBM's source instead of our comments, and every one was self-documented as deliberate; two
citations were off by 410 and 48 lines. The review wave on the deleted port found ten more in code
that had already passed `cargo check` and `cargo test` — a dropped extend bit, a wrong newtype, an
`elemOffset_` misread as an element count, and a header asserting four fields were stage-2b-only when
two are read by L3.

## What is in scope, and what is already ours

In scope, all of it type-led: `dsc/dsc2.{h,cpp}`, `dsc/designSpaceConfig.{h,cpp}`, `dsc/dims.{h,cpp}`,
`ddc/`, `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp`, and **`dsc-based-utils/DSC2ToDataflowIR/` —
bridge 1 itself**. The bridge is in the same campaign as the scheduler because it consumes the tree
the scheduler builds and the two must share ONE set of concrete node types. The deleted attempt had
them disagree: the bridge's only `ScheduleNode` was an unrelated enum while the scheduler's five node
structs carried none of the C++ base class's 13 fields.

⛔ **Do NOT re-port these — restore them from the tag instead:**
- `util/memtracker/mem_track.cpp` → was `schedule/memtrack/`, a clean 703-line port with no traits.
- the 32 `.ddl` templates and their expansion → `build.rs` generates the parser and
  `schedule/ddl/conversion.rs` mints into the tree. **The templates ARE the schedule**: expanding them
  yields 11,218 of the reference's 14,711 nodes. Do not re-derive what they already state.

⛔ **DCG/PCFG is off our path** (decided 2026-09-09). Only `DSC2ToDataflowIR` is ours, never
`PCFGToDataflowIR` or `Stitcher`.

## Crate rules that still apply

- **Pure-logic port**: no bindgen, no `-sys`, no `extern "C"`, no `unsafe`, no wrapped-layout
  `Foo`/`FooRef`/`FooMut`, whatever generic C-porting conventions say.
- **Never a runtime refusal** — `crates/compiler/deeptools/CLAUDE.md` is the crate's pre-eminent rule:
  `Err(`, `.ok_or` and `Result<` are frozen at ZERO in the DataflowIR bridge. The fact you are
  reaching for belongs in a TYPE.
- ⛔ **Never `#[should_panic(expected = "<stub>")]` on a `todo!`** — it makes a gate green while
  asserting the port is unported.
- ⛔ **No tombstone comments.** Do not narrate what a previous attempt claimed or why something was
  removed. Cite the authority line and port the body.
- ⛔ **Edit/Write only — never python or sed to modify a file.** A splice once cut 36 of 475 lines and
  reported success.
- **Newtypes, never raw scalars.** Transposing two extents must be `E0308`.
- Gate: `cargo check -p deeptools` + `cargo test -p deeptools`. Never the workspace or the acceptance
  build from an agent worktree — 6 GB of `target/` each.
- ⛔ Never commit to the branch the driver is writing.

## What "done" means

Report **per unit**, never per campaign — a campaign cannot audit its own boundary, and "1,076 units
ported" once meant 238 stubs. A unit is done with all three:

1. zero `todo!` in it, and zero traits introduced;
2. a **real non-test caller**, named;
3. every field the C++ class declares either carried or named with its `path:line` and the reason.

⭐ And the number that actually matters is not any of those: it is **how many nodes our scheduler mints
on a real program.** IBM's trees hold 14,711 nodes across the 187 `g0` fixture programs
(`~/tmp/bridge1-fixtures/g0/`, ours beside `debug/<stem>/sdsc.json`). The deleted port reached 22 of
30 on one synthetic fixture. That comparison is the gate; `cargo test` green is not.
