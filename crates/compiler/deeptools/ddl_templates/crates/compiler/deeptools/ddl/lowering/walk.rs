//! THE WALK — every statement of one program, turned into a call.
//!
//! ⛔⛔ TOTAL OVER STATEMENT KINDS BY RETURNING `Err`, NOT BY `_ => {}`. Every mnemonic either emits
//! something, is a DECLARATION whose fact went into the tables here, or names itself in the reason.
//! The interpreter's wildcard is the single line that cost the most in this port.
//!
//! ⭐⭐ AND THE PORTS BECOME NAMED LOCALS, so the `data_connect=` join happens at BUILD time. A
//! producer emits `let p_<connect> = ...`; a consumer reads it. `createDataConnectMetadata` requires
//! every connect to have a non-empty producer set (`ddcv1.cpp:3323-25`) — as generated code that is
//! Rust's own name resolution, and it cannot be faked.

use super::{
    BTreeMap, BTreeSet, OperandRef, Program, Write as _, attr_bool, attr_int, attr_str, attr_strs,
    dataflow, ints,
};
use super::rules::{
    ComputeShape, ComputeUnit, Elem, PortUnit, Unary, compute_kind,
};

/// ONE PORT — a `ddl.unit`, resolved at BUILD time.
///
/// ⭐⭐ THIS IS THE JOIN, AND IT IS `data_connect`. `createDataConnectMetadata` keys its map by the
/// `data_connect` STRING (`ddcv1.cpp:3286-3320`, `ddc_metadata.h:194`) and no producer/consumer pair
/// in any template shares an SSA name: `rope.ddl`'s `sfp_result_tensor` is `%result_sfp_fifo` (`:88`,
/// on the sfp) at one end and `%prod_compute_pe_fifo` (`:93`, on the pe) at the other. The
/// interpreter joined on the name and every lookup missed.
struct Port {
    /// `unit=` — which part of the machine this end sits on.
    unit: String,
    /// `data_connect=` — the link's identity, and the only key.
    connect: String,
    /// The `ddl.allocate` or external allocation this end is a memref over, if it is one.
    alloc: Option<u16>,
}

/// ONE PROGRAM'S LOWERING BODY, or the reason it cannot be emitted yet.
///
/// ⛔⛔ TOTAL OVER STATEMENT KINDS BY RETURNING `Err`, NOT BY `_ => {}`. Every mnemonic either emits
/// something, is a DECLARATION whose fact went into the tables above, or names itself in the reason.
/// The interpreter's wildcard is the single line that cost the most in this port.
pub fn lowering_body(program: &Program) -> Result<String, String> {
    // ⭐ THE ELEMENT TYPE IS THE BIND'S. A template serves an op-func AT A PRECISION; resolving
    // without it hands an fp16 op the fp32 kernel. Empty means ANY, which is dxp's own reading —
    // `if (!types.empty())` guards the whole check (`ddl_conversion.cpp:2137`).
    let precision = match program.formats.first() {
        Some(format) => Elem::parse(format)?,
        None => Elem::F16,
    };
    let elem = precision.rust();

    // ── the tables, resolved once: a port's unit and connect, a region's memory, the immediates ──
    let mut ports: BTreeMap<u16, Port> = BTreeMap::new();
    let mut allocs: BTreeMap<u16, String> = BTreeMap::new();
    let mut constants: BTreeSet<u16> = BTreeSet::new();
    // ⭐⭐ THE RENDEZVOUS CENSUS: per `signal_name=`, which unit signals and which waits. The two halves
    // are separate statements sharing one signal name, so this pairs them the same way the
    // `data_connect=` census pairs a wire's ends — and a signal with only one half is then visible.
    let mut syncs: BTreeMap<String, (Option<String>, Option<String>)> = BTreeMap::new();
    for (at, stmt) in program.stmts.iter().enumerate() {
        let operands = &program.operands[at];
        let one = |index: usize| -> Option<u16> {
            match operands.get(index) {
                Some(OperandRef::One(id)) => Some(*id),
                _ => None,
            }
        };
        match stmt.mnemonic.as_str() {
            "ddl.unit" => {
                let unit = attr_str(&stmt.op, "unit")
                    .ok_or_else(|| "a ddl.unit states no unit=".to_owned())?;
                let connect = attr_str(&stmt.op, "data_connect")
                    .ok_or_else(|| "a ddl.unit states no data_connect=".to_owned())?;
                for result in &program.results[at] {
                    ports.insert(
                        *result,
                        Port {
                            unit: unit.to_owned(),
                            connect: connect.to_owned(),
                            alloc: one(1),
                        },
                    );
                }
            }
            "ddl.allocate" | "ddl.get_external_data_transfer_allocation" => {
                let memory = attr_str(&stmt.op, "memory")
                    .ok_or_else(|| "an allocation states no memory=".to_owned())?;
                for result in &program.results[at] {
                    allocs.insert(*result, memory.to_owned());
                }
            }
            // ⛔ ALL THREE CONSTANT KINDS, and they are not one statement. `ddl.operand_constant`
            // names a literal, `ddl.define_constant` binds one the template computes, and
            // `ddl.get_external_constant` binds one the caller supplies — and every one of them is a
            // `SenComponents::CONSTANT`, which `createDataConnectMetadata` excludes from BOTH port
            // sets (`ddcv1.cpp:3295`, `:3304`). Missing two of the three deferred 106 programs.
            "ddl.operand_constant" | "ddl.define_constant" | "ddl.get_external_constant" => {
                for result in &program.results[at] {
                    constants.insert(*result);
                }
            }
            "ddl.sync" => {
                if let (Some(signal), Some(unit)) = (
                    attr_str(&stmt.op, "signal_name"),
                    attr_strs(&stmt.op, "units").first().copied(),
                ) {
                    // ⛔⛔ `attr_bool`. THE THIRD ATTRIBUTE READ WITH THE WRONG ACCESSOR HERE, after
                    // `rotate_num_elements` (an `I64Attr` read as a string) and `mode` (the same). Every
                    // one presented as a fact silently ABSENT rather than a mismatch: this one made all
                    // 18 signals look like sends with no receiver, so 128 programs deferred on
                    // "only one half" while the templates plainly write both.
                    let receive = attr_bool(&stmt.op, "is_receive") == Some(true);
                    let side = syncs.entry(signal.to_owned()).or_default();
                    if receive {
                        side.1 = Some(unit.to_owned());
                    } else {
                        side.0 = Some(unit.to_owned());
                    }
                }
            }
            _ => {}
        }
    }

    // ── the walk ──
    let mut body = String::new();
    let mut open: Vec<Open> = Vec::new();
    // ⭐⭐ THE ARM THAT JUST CLOSED, WAITING FOR ITS SIBLING. Two arms of one predicate are visited
    // SEPARATELY — the paths differ at that position, so the `then` closes before the `else` opens —
    // and they must become ONE `scf.if` with two regions, never two siblings on a predicate and its
    // negation (which crashed dbo-opt: "'sentient.if' op operation destroyed but still has uses").
    // ⛔ AND IT CARRIES THE LOOPS THAT WERE OPEN AROUND IT. A held arm outlives them; resolving its
    // predicate against the stack as it stands later is the bug that produced "only 0 loops enclose it".
    let mut pending: Option<(dataflow::ResolvedCond, dataflow::Arm, String, Vec<(usize, String)>)> =
        None;
    let mut arms = 0usize;
    let mut prev: &[dataflow::Enclosing] = &[];
    let mut produced: BTreeSet<String> = BTreeSet::new();
    // ⛔⛔⛔ WHICH CONNECTS THE **TEMPLATE** MAKES PREDICATE-VALUED — and nothing this walk decided.
    //
    // THIS WAS A REWARD HACK ON ITS FIRST WRITING. It recorded what the generator had ACTUALLY EMITTED
    // and then used that to decide whether a condition was legal — a check consulting its own author,
    // which by construction can never disagree. It moved the generated count 163 -> 176 by making the
    // bookkeeping agree with itself, not by getting the semantics right, and a value wire read as a lane
    // mask would have passed it silently.
    //
    // ⭐ SO EVERY ENTRY HERE COMES FROM A PARSED FACT, external to this walk:
    //   · a `ddl.compute` whose `computetype=` is a COMPARISON — its result is an `i1` by
    //     `SNComputeLowering.cpp:1618-1625`'s dispatch, which is the template's word and not ours
    //   · a port whose `ddl.allocate` names a STATE register — `PESTATE`/`SFPSTATE` hold lane masks
    //     (`:1253-1269`)
    // Nothing is inserted because the generator chose a constructor.
    let mut predicates: BTreeSet<String> = ports
        .values()
        .filter(|port| state_backed(port, &allocs))
        .map(|port| port.connect.clone())
        .collect();
    let mut viewed: BTreeSet<(String, u16)> = BTreeSet::new();
    let mut marks = 0usize;

    for (at, stmt) in program.stmts.iter().enumerate() {
        let path = &program.paths[at];

        // ⭐ THE NEST IS `stmt.path`, AND THAT IS PURE MECHANISM. Two statements share a body exactly
        // when their paths agree up to that depth, so a prefix comparison opens and closes the loops.
        let common = prev
            .iter()
            .zip(path.iter())
            .take_while(|(a, b)| same_enclosing(a, b))
            .count();
        while open.len() > common {
            match open.pop().expect("open is non-empty") {
                Open::Loop { mark, .. } => {
                    let _ = writeln!(
                        body,
                        "        e.wrap_since(&{mark}, {mark}_iv, Unresolved::owed(1));"
                    );
                }
                // ⭐ AN ARM CLOSES BY CAPTURING WHAT IT EMITTED. It is not emitted yet: its sibling
                // may be next, and the two become one `scf.if`.
                Open::Arm { mark, cond, arm, loops } => {
                    let _ = writeln!(body, "        let {mark}_arm = e.take_arm(&{mark});");
                    flush_arm(&mut body, &mut pending, loops, cond, arm, mark)?;
                }
            }
        }
        for step in path.iter().skip(common) {
            match step {
                dataflow::Enclosing::Loop { stmt } => {
                    let mark = format!("m{marks}");
                    marks += 1;
                    let _ = writeln!(body, "        let ({mark}, {mark}_iv) = e.open_loop();");
                    open.push(Open::Loop { mark, stmt: *stmt });
                }
                // ⭐⭐ AN UNDECIDED ARM IS AN `scf.if`, AND IT IS BUILT NOW. `build.rs` already
                // flattened every branch it could decide for a bind; what survives is a predicate over
                // a loop POSITION, which no walk can settle because it differs per trip. So both arms
                // are kept and paired.
                dataflow::Enclosing::Arm(of) => {
                    let mark = format!("a{arms}");
                    arms += 1;
                    let _ = writeln!(body, "        let {mark} = e.marks();");
                    // ⭐ THE ENCLOSING LOOPS ARE CAPTURED HERE, while they are still open.
                    let loops = open_loops(&open);
                    open.push(Open::Arm {
                        mark,
                        cond: of.cond.clone(),
                        arm: of.arm,
                        loops,
                    });
                }
            }
        }
        prev = path;

        let operands = &program.operands[at];
        let one = |index: usize| -> Option<u16> {
            match operands.get(index) {
                Some(OperandRef::One(id)) => Some(*id),
                Some(OperandRef::List(ids)) => ids.first().copied(),
                _ => None,
            }
        };
        let list = |index: usize| -> Vec<u16> {
            match operands.get(index) {
                Some(OperandRef::List(ids)) => ids.clone(),
                Some(OperandRef::One(id)) => vec![*id],
                _ => Vec::new(),
            }
        };

        match stmt.mnemonic.as_str() {
            // ── DECLARATIONS. They bind names the tables above already read; no DataflowIR of
            // their own. `ddl.loop` is here because the PATH carries it, not the statement.
            "ddl.unit"
            | "ddl.allocate"
            | "ddl.get_external_data_transfer_allocation"
            | "ddl.operand_constant"
            | "ddl.define_constant"
            | "ddl.get_external_constant"
            // ⛔ A CONDITION BINDS A PREDICATE AND EMITS NOTHING. `build.rs` already flattened every
            // `ddl.if` it could decide for a bind (995 exclusive branches), so a condition's only
            // consumer is a branch that is already resolved; what survives is `Enclosing::Arm`, which
            // the walk above refuses by name.
            | "ddl.condition"
            | "ddl.condition_or"
            | "ddl.condition_and"
            | "ddl.condition_not"
            | "ddl.alias_one_constant_of"
            // ⛔⛔ A LAYOUT CONSTRAINT, AND ITS FACT IS STEP 3's. `ddl.force_innermost_dimensions`
            // (`%allocation, %datastage, %dims...`) says which dims must be innermost in that
            // allocation's layout — an EXTENT fact, so it emits no op here and step 3 is what
            // consumes it. Its operands are already in `PROGRAMS`; nothing reads them yet.
            | "ddl.force_innermost_dimensions"
            // ⛔⛔ FOUR HANDLES, NO OP. `DdlOps.td:755-765` says it outright: this op "is used to
            // generate multiple handlers" — `%psum_start`/`%psum_end` are PREDICATES a `ddl.if`
            // branches on, and `%next_core`/`%prev_core` are CORE handles that appear in a
            // `ddl.unit`'s second operand slot (`:768-769`
            // `ddl.unit(%outtensor, %prev_core) {unit="sfpring", ..}`). None of the four is a value
            // this statement computes, so it emits nothing and the four names are bound by it.
            //
            // ⭐ AND THE TWO PREDICATES ARE WHY THE PSUM ARMS CANNOT BE FLATTENED. Which core is the
            // chain start is a position, not a constant the walk can settle, so those `ddl.if`s
            // survive as `Enclosing::Arm` and need `scf.if` — see the arm refusal above.
            // ⛔ A PURE TRANSFORMATION FLAG. `ddl.disable_transfer_promotion` has no operands, no
            // results and no attributes (`DdlOps.td:922-934`); its only consumer flips
            // `enableMovingDataTransfer` (`ddl_conversion.cpp:2033-2041`), which gates
            // `Ddc::hoistTransfersUpForReuse()` (`ddcv1.cpp:3760-3763`). It creates no schedule node,
            // so nothing reaches DataflowIR.
            | "ddl.disable_transfer_promotion"
            | "ddl.core_to_core_communication"
            // ⛔ A PARAMETRIC LOOP IS STILL A LOOP, and the PATH carries it. Same as `ddl.loop`: the
            // statement itself emits nothing because `stmt.path` is what opens the region.
            | "ddl.parametric_loop"
            | "ddl.dimension"
            | "ddl.padded_dimension"
            | "ddl.layout"
            | "ddl.tensor"
            | "ddl.internal_tensor"
            | "ddl.type"
            | "ddl.datastage"
            | "ddl.get_external_datastage"
            | "ddl.datastage_constraint"
            | "ddl.constraint"
            | "ddl.operation_bind"
            | "ddl.loop"
            | "ddl.alias_one_tensor_of"
            | "ddl.dataflow"
            | "ddl.transformations" => {}

            "ddl.data_transfer" => {
                let src = one(0).ok_or_else(|| "a transfer names no source".to_owned())?;
                let dst = one(1).ok_or_else(|| "a transfer names no destination".to_owned())?;
                let src_port = ports
                    .get(&src)
                    .ok_or_else(|| "a transfer's source is not a ddl.unit".to_owned())?;
                let dst_port = ports
                    .get(&dst)
                    .ok_or_else(|| "a transfer's destination is not a ddl.unit".to_owned())?;
                let dst = PortUnit::parse(&dst_port.unit)?;
                let (dst_unit, dst_kind) = (dst.local(), dst.link());
                let dst_dfir = dst.dfir();

                // ⛔⛔ A SOURCE ON `unit="constant"` IS AN IMMEDIATE, NOT A WIRE END, and 66 programs
                // state one. `createDataConnectMetadata` skips `SenComponents::CONSTANT` on both the
                // consumer side (`ddcv1.cpp:3295`) and the producer side (`:3304`), so there is no
                // link to build and nothing to send: the destination simply holds the literal.
                //
                // ⭐ SO THE HANDLE IS BOUND PER BRANCH. This arm names its unit by KIND and has no use
                // for the `Val`, while the transfer below spends both ends — binding it above the
                // branch made every constant-sourced transfer an unused-variable warning in generated
                // code, which is where a real warning would hide.
                if src_port.unit == "constant" {
                    let _ = writeln!(
                        body,
                        "        e.get_unit({dst_dfir});\n        \
                         let p_{} = e.constant({dst_dfir}, 0, Emit::wire_of::<A>({elem}));",
                        dst_port.connect
                    );
                    produced.insert(dst_port.connect.clone());
                } else {
                    let _ = writeln!(
                        body,
                        "        let u_{dst_unit} = e.get_unit({dst_dfir});"
                    );
                    let src = PortUnit::parse(&src_port.unit)?;
                    let (src_unit, src_kind) = (src.local(), src.link());
                    let src_dfir = src.dfir();
                    let _ = writeln!(
                        body,
                        "        let u_{src_unit} = e.get_unit({src_dfir});"
                    );

                    // ⭐ THE SOURCE'S DATA: a load of its region if this end IS one, else what the
                    // producer of its `data_connect=` bound.
                    //
                    // ⛔ HOISTED INTO ITS OWN `let`, because `e.transfer(.., e.vector_load(..), ..)`
                    // borrows the emitter twice and E0499 is the whole generated module at once.
                    let data = source_data(
                        &mut body,
                        src_port,
                        &allocs,
                        &produced,
                        &mut viewed,
                        src,
                        &elem,
                        // ⛔ AN `I64Attr`, NOT A STRING. `OptionalAttr<I64Attr>:$rotate_num_elements`
                        // (`DdlOps.td:605`), and reading it with `attr_str` returned `None` for every
                        // transfer — the rotation silently absent, which is the defect it encodes.
                        attr_int(&stmt.op, "rotate_num_elements"),
                    )?;
                    // ⛔⛔ A TRANSFER INTO A STATE REGISTER BINDS A PREDICATE. `PESTATE`/`SFPSTATE`
                    // hold `i1`s — `constructFMINorFMAXOperation` puts a COMPARE's `bool_type` result
                    // there and the SELECTION's everywhere else (`SNComputeLowering.cpp:1253-1269`) —
                    // so a `SELECT` reading one is reading its condition. Deciding it from the
                    // DESTINATION'S OWN MEMORY is what avoids a `Wire -> Predicate` conversion, which
                    // would let any value be spent as a condition and reopen the `%24#0` hole.
                    let into_state = dst_port
                        .alloc
                        .and_then(|region| allocs.get(&region))
                        .is_some_and(|memory| matches!(memory.as_str(), "sfpstate" | "pestate"));
                    let carry = if into_state {
                        "transfer_predicate"
                    } else {
                        "transfer"
                    };
                    let _ = writeln!(
                        body,
                        "        let d_{} = {data};\n        \
                         let p_{} = e.{carry}::<link::{src_kind}, link::{dst_kind}>(u_{src_unit}, u_{dst_unit}, d_{}, Emit::wire_of::<A>({elem}));",
                        dst_port.connect, dst_port.connect, dst_port.connect
                    );
                    produced.insert(dst_port.connect.clone());
                }

                // ⭐ AND IF THE DESTINATION IS A MEMREF END, the arrival is stored through its view.
                if let Some(region) = dst_port.alloc {
                    if allocs.get(&region).map(String::as_str) == Some("lx") {
                        let view = view_of(&mut body, &mut viewed, dst, region, &elem);
                        let _ = writeln!(
                            body,
                            "        e.vector_store({dst_dfir}, p_{}, {view});",
                            dst_port.connect
                        );
                    }
                }
            }

            "ddl.compute" => {
                let computetype = attr_str(&stmt.op, "computetype")
                    .ok_or_else(|| "a ddl.compute states no computetype=".to_owned())?;
                let unit = attr_str(&stmt.op, "unit")
                    .ok_or_else(|| "a ddl.compute states no unit=".to_owned())?;
                // ⛔⛔ A COMPUTE'S UNIT MAY BE A ROW SPAN, AND THE REFERENCE CLONES THE NODE PER ROW.
                // `ddl_conversion.cpp:730-731` rewrites `"pt"` to `"ptrow0-{N-1}"`, `unrollRowUnits`
                // pushes one component per row (`:734-749`), and `:1467-1472` copies the compute for
                // each. So this is not one unit but a scope over the arch's rows.
                let on_unit = ComputeUnit::parse(unit)?;
                let (scope_open, on, scope_close) = on_unit.render();
                body.push_str(&scope_open);
                // ⛔ `attr_int`, NOT `attr_str`. `mode=` is `OptionalAttr<I64Attr>` (`DdlOps.td:625-631`)
                // and reading it as a string gave `None` for every template — the second time an
                // attribute here was read with the wrong accessor, after `rotate_num_elements`.
                let kind = compute_kind(
                    computetype,
                    attr_int(&stmt.op, "mode"),
                    attr_int(&stmt.op, "sign_extend"),
                    precision,
                )?;
                let ins = list(0);
                let dst = one(1).ok_or_else(|| "a compute names no destination".to_owned())?;
                let dst_port = ports
                    .get(&dst)
                    .ok_or_else(|| "a compute's destination is not a ddl.unit".to_owned())?;
                // ⛔ CALLED FOR ITS EFFECT, NOT ITS VALUE. A compute names its unit by KIND
                // (`e.mac(DfirUnit::Sfp, ..)`), so the handle is unused here — but the call still has
                // to happen, because it is what puts the `dataflow.get_unit` in the preamble and a
                // `dataflow.program_unit` needs that `Val` for its `on`. Binding it would be an
                // unused-variable warning in generated code, which is where a real warning would hide.
                let _ = writeln!(body, "        e.get_unit({on});");

                // ⭐⭐ EVERY OPERAND IS RESOLVED BY `data_connect=`, HERE. An input that is an
                // immediate is one; otherwise it is the local its producer bound — and if no
                // producer bound one, the generated code names an unbound local and the build stops,
                // which is `ddcv1.cpp:3323-25` enforced by name resolution.
                // ⛔ EVERY ARGUMENT IS HOISTED INTO ITS OWN `let`. A constructor call nested in
                // another borrows the emitter twice, and E0499 lands on the whole generated module at
                // once — 19 of them from `e.mac(.., e.constant(..), ..)` alone.
                let mut args = Vec::new();
                let mut immediate = 0usize;
                for (index, input) in ins.iter().enumerate() {
                    let port = ports.get(input);
                    let is_immediate = constants.contains(input)
                        || port.is_some_and(|port| port.unit == "constant");
                    if is_immediate {
                        // ⛔ A PORT ON `unit="constant"` IS AN IMMEDIATE TOO — `ddcv1.cpp:3295` skips
                        // it on the consumer side, so it never joins a `data_connect=`.
                        let name = format!("c{immediate}");
                        immediate += 1;
                        let _ = writeln!(
                            body,
                            "        let {name} = e.constant({on}, 0, Emit::wire_of::<A>({elem}));"
                        );
                        args.push(name);
                        continue;
                    }
                    let port = port.ok_or_else(|| {
                        "a compute operand is neither a ddl.unit nor a constant".to_owned()
                    })?;
                    // ⛔⛔ AND ITS `data_connect=` MUST HAVE A PRODUCER. The check existed for a
                    // TRANSFER's source and not for a compute's operands, so 19 programs emitted
                    // `p_<connect>` for a connect nothing bound — unbound locals, which is
                    // `ddcv1.cpp:3323-25`'s "producer set must not be empty" caught by Rust's name
                    // resolution. Right diagnosis, wrong stage: a program that cannot be built must
                    // DEFER with the reason, not break the generated module.
                    // ⛔⛔ A REGISTER-FILE OPERAND IS A READ, NOT A WIRE — and the systolic accumulator
                    // is why. `bmm.ddl`'s PT MAC takes `%arf_ptsum` as its accumulator AND writes it,
                    // one `data_connect` accumulating across rows in the ARF. As a wire that is a local
                    // read before it is bound; as a register read it is the file's contents.
                    //
                    // ⭐ AND THE CONDITION IS SELF-ACCUMULATION, NOT A MEMORY NAME: the operand's
                    // `data_connect` IS the destination's. Testing the allocation's `memory=` instead
                    // missed it, because the accumulator port need not carry one — what identifies the
                    // case is that the compute reads what it is about to write.
                    let accumulates_into_itself = port.connect == dst_port.connect;
                    // ⛔⛔ AND A REGISTER-FILE OR PEER-CORE OPERAND IS ALSO A READ, NOT A MISSING
                    // PRODUCER. Three shapes reach a compute without any statement in THIS program
                    // having bound them:
                    //
                    //  · the systolic accumulator, above — reads what it writes
                    //  · a register file (`ddl.allocate {memory="sfplrf"|"pelrf"|"ptarf"|..}`) — the ddc
                    //    places it into that file's own tracker and `finalizeOps` names it
                    //    `R<addr/bytesPerStick>` (`ddcv1.cpp:183-360`, `:3345-3392`); nothing sends to it
                    //  · a PEER CORE's output. `core_to_core_communication` hands out `%prev_core`
                    //    exactly so a port can name it (`DdlOps.td:768`,
                    //    `ddl.unit(%outtensor, %prev_core) {unit="sfpring", ..}`), and the producer is
                    //    the NEIGHBOUR's program, not this one.
                    //
                    // Treating any of them as an empty producer set is reading `ddcv1.cpp:3323-25` too
                    // literally: that check is over the whole schedule, not over one op-bind's walk.
                    let in_register_file = port
                        .alloc
                        .and_then(|region| allocs.get(&region))
                        .is_some_and(|memory| {
                            matches!(
                                memory.as_str(),
                                "sfplrf" | "pelrf" | "ptarf" | "ptxrf" | "sfpstate" | "pestate"
                            )
                        });
                    let from_a_peer = port.unit == "sfpring";
                    if (accumulates_into_itself || in_register_file || from_a_peer)
                        && !produced.contains(&port.connect)
                    {
                        let name = format!("r{immediate}");
                        immediate += 1;
                        // ⛔ A SELECT'S FIRST OPERAND IS ITS CONDITION, and a condition read from a
                        // state register is a PREDICATE — `PESTATE`/`SFPSTATE` are where the reference
                        // puts a compare's i1 (`SNComputeLowering.cpp:1253-1269`). Reading it as a value
                        // wire is an `i1` spent where a vector belonged.
                        // ⛔⛔ A REGISTER-FILE OPERAND IS AN ADDRESS, A VIEW AND A LOAD — not a bare
                        // constant, which is what this emitted and which the reference never emits.
                        // `constructComputeInputOperandAndAddToList`'s register-file branch
                        // (`SNComputeLowering.cpp:521-594`) builds the address as
                        // `arith.constant startAddr * getAddressGranularityMultiplyFactor(..)`
                        // (`:536-537`, `:562-565`), a `get_logical_memory_view` over it (`:579-586`),
                        // then an `agen.vector_load` of that view (`:590-594`).
                        //
                        // ⛔ AND THE ADDRESS IS OWED, ONLY THE ADDRESS. `startAddr` is the operand's own
                        // `data_info->startAddr_` and the factor is the arch's granularity — step 2's
                        // both, so the view takes `Addr::owed()` like every other view here.
                        let _ = writeln!(
                            body,
                            "        let lx = e.lx();\n        \
                             let {name} = e.register_read({on}, memory::Base::new(lx, memory::Addr::owed()), &Tile::owed(), {elem}, Emit::wire_of::<A>({elem}));"
                        );
                        args.push(name);
                        continue;
                    }
                    if !produced.contains(&port.connect) {
                        return Err(format!(
                            "`data_connect=\"{}\"` is read by a compute with no producer — \
                             `ddcv1.cpp:3323-25` refuses an empty producer set",
                            port.connect
                        ));
                    }
                    args.push(format!("p_{}", port.connect));
                }

                // ⭐⭐⭐ ONE MATCH ON A VARIANT, AND NO STRING PROTOCOL. This dispatch once received
                // `"estimate:Exp:A"` / `"binary:Max"` from `compute_kind` and took them apart with
                // `strip_prefix` and `split_once` — two functions in one crate talking over encoded
                // text, where the encoder and the parser could disagree and nothing would say so.
                // `ComputeShape` is the closed set; text survives only in the `rust()` renderers.
                //
                // ⛔ AND THE ARITY IS CHECKED PER SHAPE, WHICH THE REFERENCE DOES NOT DO. Its guard is
                // `is_any_of(inputs_.size(), 2, 3)` (`SNComputeLowering.cpp:1038-1040`) and then each
                // branch hard-codes which operands it reads — so a `PACKMERGE` written with three
                // inputs has its third SILENTLY IGNORED, and a `SELECT` written with two reads
                // `inputs[2]` OUT OF BOUNDS with no diagnostic. Silent acceptance is the whole failure
                // this generator exists to remove.
                let need = |want: usize| -> Result<(), String> {
                    if args.len() == want {
                        Ok(())
                    } else {
                        Err(format!(
                            "computetype=\"{computetype}\" takes {want} operands but the template \
                             writes {} — the reference would accept this silently",
                            args.len()
                        ))
                    }
                };
                match kind {
                    ComputeShape::Mac => {
                        need(3)?;
                        let _ = writeln!(
                            body,
                            "        let p_{} = e.mac({on}, {}, {}, {});",
                            dst_port.connect, args[0], args[1], args[2]
                        );
                    }
                    ComputeShape::Binary(op) => {
                        need(2)?;
                        let _ = writeln!(
                            body,
                            "        let p_{} = e.binary({on}, BinaryOp::{}, {}, {});",
                            dst_port.connect,
                            op.rust(),
                            args[0],
                            args[1]
                        );
                    }
                    ComputeShape::Compare(op) => {
                        need(2)?;
                        let _ = writeln!(
                            body,
                            "        let p_{} = e.compare({on}, CompareOp::{}, {}, {});",
                            dst_port.connect,
                            op.rust(),
                            args[0],
                            args[1]
                        );
                        // ⭐ A PARSED FACT: the template said `computetype=` is a comparison, so this
                        // connect carries an `i1`. Not "we chose to emit a compare".
                        predicates.insert(dst_port.connect.clone());
                    }
                    // ⛔⛔ TWO OPS, AND WHICH RESULT LEAVES IS THE DESTINATION'S TO SAY.
                    // `constructFMINorFMAXOperation` (`:1240-1251`) builds an `ElementWiseCompareOp`
                    // then an `ElementWiseSelectionOp` over `(compare, inputs[0], inputs[1])`; a
                    // `PESTATE`/`SFPSTATE` output takes the i1 COMPARE and anything else the SELECTION
                    // (`:1253-1269`).
                    //
                    // ⭐ NO VENDORED TEMPLATE STATES A STATE-REGISTER DESTINATION — the `unit=` census
                    // has no `pestate`/`sfpstate` — so the selection always leaves, and that assumption
                    // is PROVED by the refusal below rather than taken silently.
                    ComputeShape::MinMax(op) => {
                        need(2)?;
                        if dst_port.unit == "sfpstate" || dst_port.unit == "pestate" {
                            return Err(format!(
                                "computetype=\"{computetype}\" writing to unit=\"{}\" takes the i1 \
                                 COMPARE rather than the selection \
                                 (SNComputeLowering.cpp:1253-1269), which this arm does not emit",
                                dst_port.unit
                            ));
                        }
                        let cond = format!("k{}", dst_port.connect);
                        let _ = writeln!(
                            body,
                            "        let {cond} = e.compare({on}, CompareOp::{}, {}, {});\n        \
                             let p_{} = e.select({on}, {cond}, {}, {});",
                            op.rust(),
                            args[0],
                            args[1],
                            dst_port.connect,
                            args[0],
                            args[1]
                        );
                    }
                    // ⛔ THE FIRST OPERAND IS THE CONDITION, not a value (`:1170-1174` passes
                    // `inputs[0]` where `element_wise_selection` takes `cond`).
                    ComputeShape::Select => {
                        need(3)?;
                        // ⛔⛔ THE CONDITION MUST BE A PREDICATE, AND A PRODUCED CONNECT IS A `Wire`.
                        // A select whose condition comes from a state register is reading an `i1`
                        // (`SNComputeLowering.cpp:1253-1269` puts a compare's result there), but a
                        // connect produced by a transfer or a compute is typed as a value wire. Closing
                        // that needs the PORT to carry whether its connect is predicate-valued — a
                        // per-connect type, which the parse can supply from the allocation's `memory=`.
                        //
                        // ⛔ WHAT IT MUST NOT BE IS A `Wire -> Predicate` CONVERSION. That hole is
                        // precisely what let a memref be spent as a MAC operand and produced the
                        // `%24#0` refusal; one program is not worth reopening it.
                        // ⭐ THE CONDITION IS A PREDICATE BY CONSTRUCTION NOW: it is either a
                        // `state_read` or a connect that ARRIVED in a state register via
                        // `transfer_predicate`. Both bind a `Predicate`, so the type checks rather than
                        // being asserted — and a condition that is neither fails to compile at the
                        // `e.select` call, naming the program.
                        // ⛔ AND ITS DESTINATION DECIDES WHAT IT BINDS. A select writing a state
                        // register yields the `i1`, so the next select can read it as a condition —
                        // same rule as FMIN/FMAX's (`SNComputeLowering.cpp:1253-1269`).
                        // ⛔ A CONDITION MUST BE ONE THIS WALK BOUND AS A PREDICATE. Anything else is
                        // a value wire being read as a lane mask, and the fix is never a conversion —
                        // that is the hole the `%24#0` refusal came through.
                        if !predicates.contains(&args[0].trim_start_matches("p_").to_owned())
                            && !args[0].starts_with('r')
                        {
                            return Err(format!(
                                "computetype=\"SELECT\" reads its condition from `{}`, which this walk \
                                 bound as a value wire — a condition is an `i1` \
                                 (SNComputeLowering.cpp:1253-1269), and the destination that would make \
                                 it one is not a state register here",
                                args[0]
                            ));
                        }
                        // ⛔ AND ITS OWN DESTINATION DECIDES WHAT IT BINDS: writing a state register
                        // yields the i1, so a later select can read it — the FMIN/FMAX rule again.
                        // ⛔ FROM THE PARSE ONLY. This once also tested `predicates.contains(dst)` and
                        // then INSERTED it — marking a connect a predicate because it had been marked
                        // one. Self-fulfilling, and it is what inflated the generated count.
                        let call = if state_backed(dst_port, &allocs) {
                            "select_into_state"
                        } else {
                            "select"
                        };
                        let _ = writeln!(
                            body,
                            "        let p_{} = e.{call}({on}, {}, {}, {});",
                            dst_port.connect, args[0], args[1], args[2]
                        );
                    }
                    ComputeShape::Pack => {
                        need(2)?;
                        let indices = ints(&stmt.op, "indices");
                        if indices.is_empty() {
                            return Err(
                                "computetype=\"PACKMERGE\" states no indices= mapping".to_owned()
                            );
                        }
                        let repetition = attr_int(&stmt.op, "repetition").unwrap_or(8);
                        let sign_extend = attr_int(&stmt.op, "sign_extend") == Some(1);
                        let _ = writeln!(
                            body,
                            "        let p_{} = e.pack({on}, {}, {}, vec![{indices}], {repetition}, \
                             {sign_extend}, {elem});",
                            dst_port.connect, args[0], args[1]
                        );
                    }
                    ComputeShape::Unary(unary) => {
                        need(1)?;
                        let call = match unary {
                            Unary::Floor => format!("e.floor({on}, {})", args[0]),
                            Unary::Reduce(op) => format!(
                                "e.reduce({on}, BinaryOp::{}, {})",
                                op.rust(),
                                args[0]
                            ),
                            // ⛔ THE INDEX COUNT IS THE ELEMENT TYPE'S — eight lanes for f16, four for
                            // f32 (`SNComputeLowering.cpp:1406-1451`). An f32 splat with eight indices
                            // describes a pattern twice the register's width.
                            Unary::Splat { sign_extend } => {
                                let lanes = match Elem::parse(
                                    program.formats.first().map_or("Sen169Fp16", String::as_str),
                                )? {
                                    Elem::F32 | Elem::Int(32) => 4,
                                    _ => 8,
                                };
                                let indices: Vec<String> = (0..lanes)
                                    .map(|lane| {
                                        if sign_extend && lane % 2 == 1 {
                                            "-1".to_owned()
                                        } else {
                                            "0".to_owned()
                                        }
                                    })
                                    .collect();
                                // ⛔ ABSENT MEANS THE DEFAULT, NOT MISSING. `repetition_` is 8
                                // (`dsc2.h:907`), which is `Arch::SLICES_PER_STICK` — the same default
                                // PACKMERGE takes. Requiring it deferred all 10 SPLAT programs on a
                                // value the reference supplies itself.
                                let repetition = attr_int(&stmt.op, "repetition").unwrap_or(8);
                                format!(
                                    "e.shuffle({on}, {}, vec![{}], {repetition})",
                                    args[0],
                                    indices.join(", ")
                                )
                            }
                            Unary::FastExp => format!("e.fast_exp({on}, {})", args[0]),
                            // ⛔ THE MAPPING AND THE REPEAT ARE THE TEMPLATE'S. `repetition=` is 8
                            // throughout the vendored set — `Arch::SLICES_PER_STICK` — and is carried
                            // rather than assumed so a template stating otherwise is visible.
                            Unary::Shuffle => {
                                let indices = ints(&stmt.op, "indices");
                                if indices.is_empty() {
                                    return Err(
                                        "computetype=\"SHUFFLE\" states no indices= mapping"
                                            .to_owned(),
                                    );
                                }
                                let repetition =
                                    attr_int(&stmt.op, "repetition").ok_or_else(|| {
                                        "computetype=\"SHUFFLE\" states no repetition=".to_owned()
                                    })?;
                                format!(
                                    "e.shuffle({on}, {}, vec![{indices}], {repetition})",
                                    args[0]
                                )
                            }
                            // ⛔ THE KIND AND ITS VERSION TRAVEL TOGETHER, because `rec`/`ln` take no
                            // version while `exp`/`sigmoid`/`tanh` do, and the MODE decides which —
                            // see `rules::compute_kind`'s FEST table, which read `Exp` for every mode
                            // and would have emitted an exponential for a reciprocal.
                            Unary::Estimate(est, version) => {
                                let version = match version {
                                    Some(ver) => format!("Some(EstimateVersion::{})", ver.rust()),
                                    None => "None".to_owned(),
                                };
                                format!(
                                    "e.estimate({on}, EstimateKind::{}, {version}, {})",
                                    est.rust(),
                                    args[0]
                                )
                            }
                        };
                        let _ = writeln!(body, "        let p_{} = {call};", dst_port.connect);
                    }
                }
                produced.insert(dst_port.connect.clone());
                body.push_str(&scope_close);
            }

            // ⭐⭐ ONE HALF OF A RENDEZVOUS, AND THE SIGNAL NAME IS THE PAIRING. The templates write
            // the two halves as separate statements on one `signal_name=` — `{units=["lxsu"],
            // is_receive=false, ..}` then `{units=["lxlu"], is_receive=true, ..}` — so the peer is the
            // other statement's unit, resolved from the census below exactly as a wire's endpoints are
            // resolved from `data_connect=`.
            //
            // ⛔ A SIGNAL WITH ONLY ONE HALF IS REFUSED. `dataflow.sync_recv` is BLOCKING
            // (`Dataflow.td:209-212`), so a wait whose signaller never runs is a unit that hangs on the
            // card — and that is the failure this crate's notes record as `CB state=TimedOut`.
            "ddl.sync" => {
                let signal = attr_str(&stmt.op, "signal_name")
                    .ok_or_else(|| "a ddl.sync states no signal_name=".to_owned())?;
                let receive = attr_bool(&stmt.op, "is_receive") == Some(true);
                let mine = crate::ident_of(
                    attr_strs(&stmt.op, "units")
                        .first()
                        .ok_or_else(|| "a ddl.sync names no units=".to_owned())?,
                );
                let peer = syncs
                    .get(signal)
                    .and_then(|(sender, waiter)| {
                        if receive {
                            sender.as_deref()
                        } else {
                            waiter.as_deref()
                        }
                    })
                    .ok_or_else(|| {
                        format!(
                            "signal_name=\"{signal}\" has only one half — `dataflow.sync_recv` is \
                             blocking (Dataflow.td:209-212), so an unmatched wait hangs the unit"
                        )
                    })?;
                let on_port = PortUnit::parse(&mine.to_ascii_lowercase())?;
                let (on, on_dfir) = (on_port.local(), on_port.dfir());
                let peer_port = PortUnit::parse(&peer.to_ascii_lowercase())?;
                let (peer_unit, peer_dfir) = (peer_port.local(), peer_port.dfir());
                let _ = writeln!(
                    body,
                    "        let u_{on} = e.get_unit({on_dfir});\n        \
                     let u_{peer_unit} = e.get_unit({peer_dfir});\n        \
                     e.sync({on_dfir}, u_{peer_unit}, SyncSignal::{}, {receive});",
                    crate::ident_of(signal)
                );
                let _ = &on;
            }

            // ⭐⭐ `ddl.implicit_sync` IS A DIFFERENT OP FROM `ddl.sync`, and its units are AUTOMATIC.
            // `ddl_conversion.cpp:1781-1788` sets `units_ = {L0SU} ∪ {L0LUROW0..numPTRows-1}` and names
            // it `"sync_implicit_L0"` — the DDL statement carries no units, no direction and no signal,
            // just the allocation. `constructImplicitSyncOperation` then emits ONE op per side, each
            // naming the other as `dst` (`SNSyncLowering.cpp:141-146`).
            //
            // ⛔ ONLY L0SU AND L0LUROW0 GET A PROGRAM. Rows 1-7 are folds of row 0 and `constructUnits`
            // skips them (`:26-27`), while `constructImplicitSyncOperation` `DT_CHECK`s
            // `comp_ ∈ {L0SU, L0LUROW0}` — so emitting for the other rows trips the check.
            "ddl.implicit_sync" => {
                let l0su = PortUnit::L0su;
                let l0lu = PortUnit::L0lu;
                let _ = writeln!(
                    body,
                    "        let u_{} = e.get_unit({});\n        \
                     let u_{} = e.get_unit({});\n        \
                     e.implicit_sync({}, u_{}, Unresolved::owed(1), {elem});\n        \
                     e.implicit_sync({}, u_{}, Unresolved::owed(1), {elem});",
                    l0su.local(),
                    l0su.dfir(),
                    l0lu.local(),
                    l0lu.dfir(),
                    l0su.dfir(),
                    l0lu.local(),
                    l0lu.dfir(),
                    l0su.local(),
                );
            }

            other => return Err(format!("`{other}` has no emitter arm")),
        }
    }

    // ⛔ EVERYTHING STILL OPEN CLOSES, OUTERMOST LAST. A construct left open is a region whose body
    // was emitted flat beside it rather than inside it.
    while let Some(entry) = open.pop() {
        match entry {
            Open::Loop { mark, .. } => {
                let _ = writeln!(
                    body,
                    "        e.wrap_since(&{mark}, {mark}_iv, Unresolved::owed(1));"
                );
            }
            Open::Arm { mark, cond, arm, loops } => {
                let _ = writeln!(body, "        let {mark}_arm = e.take_arm(&{mark});");
                flush_arm(&mut body, &mut pending, loops, cond, arm, mark)?;
            }
        }
    }
    // ⛔ AND A HELD ARM WHOSE SIBLING NEVER CAME IS STILL A BRANCH. Dropping it would emit its
    // statements unconditionally, which is the template saying "maybe" and us saying "always".
    if let Some((cond, arm, mark, loops)) = pending.take() {
        emit_one_armed(&mut body, &cond, arm, &mark, &loops)?;
    }

    // ⛔ A PROGRAM THAT EMITTED NOTHING IS NOT A LOWERING. `rope.ddl`'s two binds each have a
    // dataflow region; a walk that produced no call means the region was not reached.
    if body.trim().is_empty() {
        return Err("the walk emitted no DataflowIR at all".to_owned());
    }
    Ok(body)
}

/// WHAT A TRANSFER'S SOURCE HANDS OVER.
/// ⛔⛔ AND IT CARRIES THE TRANSFER'S ROTATION, WHICH GOES ON THIS SIDE.
/// `SNTransferLowering.cpp:2237-2245` builds the `vectorchain::RotateOp` immediately after the
/// `agen.vector_load` and REPLACES the load's result with it (`load_op_result = rot_op.getResult()`),
/// under `DT_CHECK_MSG(comp_ == LXLU, "Rotation is allowed only in LXLU")`. `ddl_conversion.cpp`
/// enforces the same at ingestion — *"Attribute rotate_num_elements was specified on the
/// data_transfer operation. The source must be LXLU"* (`:1231-1236`) — so a rotation on any other
/// source is illegal ddl, and this refuses it by name rather than emitting it somewhere plausible.
fn source_data(
    body: &mut String,
    port: &Port,
    allocs: &BTreeMap<u16, String>,
    produced: &BTreeSet<String>,
    viewed: &mut BTreeSet<(String, u16)>,
    port_unit: PortUnit,
    elem: &str,
    rotate: Option<i64>,
) -> Result<String, String> {
    // ⛔ THE UNIT TRAVELS AS A VARIANT, not as two derived strings. Its LOCAL name and its `DfirUnit`
    // PATH are different renderings of one fact, and passing them separately is how a `u_DfirUnit::..`
    // gets emitted.
    let unit = port_unit.local();
    let dfir = port_unit.dfir();
    if let Some(by) = rotate {
        if port_unit != PortUnit::Lxlu {
            return Err(format!(
                "rotate_num_elements= on a transfer whose source is {unit}, but the source must be \
                 LXLU (ddl_conversion.cpp:1231-1236, SNTransferLowering.cpp:2238)"
            ));
        }
        let Some(region) = port.alloc else {
            return Err(
                "rotate_num_elements= on a transfer whose source is not a memref end".to_owned(),
            );
        };
        if allocs.get(&region).map(String::as_str) != Some("lx") {
            return Err("rotate_num_elements= on a source that is not an lx region".to_owned());
        }
        let view = view_of(body, viewed, port_unit, region, elem);
        let loaded = format!("r{region}");
        let _ = writeln!(
            body,
            "        let {loaded} = e.vector_load({dfir}, {view}, Emit::wire_of::<A>({elem}));"
        );
        // ⭐ THE ROTATE REPLACES THE LOAD'S RESULT, exactly as the reference does — the send carries
        // the rotated vector and nothing downstream sees the unrotated one.
        return Ok(format!("e.rotate({dfir}, {loaded}, {by})"));
    }
    if let Some(region) = port.alloc {
        if allocs.get(&region).map(String::as_str) == Some("lx") {
            let view = view_of(body, viewed, port_unit, region, elem);
            return Ok(format!(
                "e.vector_load({dfir}, {view}, Emit::wire_of::<A>({elem}))"
            ));
        }
    }
    // ⛔ NOT A MEMREF END, SO THE DATA IS ITS `data_connect=`'s. A register-file region has no
    // scratchy address — the backend assigns it — so what flows is the wire, not a view.
    if produced.contains(&port.connect) {
        return Ok(format!("p_{}", port.connect));
    }
    Err(format!(
        "`data_connect=\"{}\"` is read with no producer — `ddcv1.cpp:3323-25` refuses an empty producer set",
        port.connect
    ))
}

/// ONE VIEW PER (REGION, UNIT), emitted once and named after both.
fn view_of(
    body: &mut String,
    viewed: &mut BTreeSet<(String, u16)>,
    port_unit: PortUnit,
    region: u16,
    elem: &str,
) -> String {
    let unit = port_unit.local();
    let dfir = port_unit.dfir();
    let name = format!("v_{unit}_{region}");
    if viewed.insert((unit.to_owned(), region)) {
        let _ = writeln!(
            body,
            "        let lx = e.lx();\n        \
             let {name} = e.view({dfir}, memory::Base::new(lx, memory::Addr::owed()), &Tile::owed(), {elem});"
        );
    }
    format!("{name}.clone()")
}

/// WHAT IS CURRENTLY OPEN AROUND THE STATEMENTS BEING EMITTED.
///
/// ⛔ A LOOP AND AN ARM CLOSE DIFFERENTLY — a loop wraps in place, an arm is captured and held for its
/// sibling — so they cannot share one stack of names.
enum Open {
    /// A `ddl.loop`: its marks and iv are bound under `{mark}` / `{mark}_iv`.
    Loop {
        /// The generated local's name.
        mark: String,
        /// ⛔ WHICH LOOP — the index of its own statement, which is its identity and what a predicate
        /// names. See `Cond::Position`.
        stmt: usize,
    },
    /// One arm of an undecided `ddl.if`.
    Arm {
        /// The generated local's name.
        mark: String,
        /// Its predicate, loop labels already resolved to depths.
        cond: dataflow::ResolvedCond,
        /// Which side.
        arm: dataflow::Arm,
        /// ⛔⛔ THE LOOPS OPEN AROUND THIS ARM, BY IDENTITY, CAPTURED WHEN IT OPENED. Two things were
        /// wrong at once: the predicate named a loop by POSITION (fixed by carrying the statement
        /// index), and the emitter resolved it against the CURRENT stack — but an arm is HELD in
        /// `pending` until its sibling arrives, and by the final drain the stack is empty. That is what
        /// produced "a guard names the loop at depth 3 but only 0 loops enclose it".
        loops: Vec<(usize, String)>,
    },
}

/// WHETHER A PORT IS BACKED BY A STATE REGISTER — a PARSED fact, from the allocation's `memory=`.
///
/// ⛔ `PESTATE`/`SFPSTATE` HOLD LANE MASKS. `constructFMINorFMAXOperation` sends a COMPARE's
/// `bool_type` result to a state output and the SELECTION's to anything else
/// (`SNComputeLowering.cpp:1253-1269`), so the destination component is what decides whether a compute
/// yields an `i1`. That is the template's word, which is why this reads the allocation and never the
/// generator's own bookkeeping.
fn state_backed(port: &Port, allocs: &BTreeMap<u16, String>) -> bool {
    port.alloc
        .and_then(|region| allocs.get(&region))
        .is_some_and(|memory| matches!(memory.as_str(), "sfpstate" | "pestate"))
        || port.unit == "sfpstate"
        || port.unit == "pestate"
}

/// THE LOOPS CURRENTLY OPEN, BY IDENTITY AND MARK.
///
/// ⛔ ARMS DO NOT COUNT. A predicate names a LOOP, and the arms interleaved with those loops are not
/// loops — `walk` in `ddl/dataflow.rs` increments `depth` only for a `ddl.loop`, which is the same
/// statement.
fn open_loops(open: &[Open]) -> Vec<(usize, String)> {
    open.iter()
        .filter_map(|entry| match entry {
            Open::Loop { mark, stmt } => Some((*stmt, mark.clone())),
            Open::Arm { .. } => None,
        })
        .collect()
}

/// PAIR AN ARM WITH ITS SIBLING, OR HOLD IT UNTIL THE SIBLING ARRIVES.
///
/// ⛔⛔ ONE `scf.if` WITH TWO REGIONS, NEVER TWO SIBLINGS ON A PREDICATE AND ITS NEGATION. The
/// negated form crashed the backend once the predicate was compound: dbo-opt converted the shared
/// `arith.andi` into a `sentient.if`, destroyed it converting the guard, and died on the `xori` still
/// holding it — *"'sentient.if' op operation destroyed but still has uses"*.
///
/// ⛔ AND A NEGATION IS NOT A PREDICATE OPERATOR. `ddl.condition_not` swaps which arm the statements
/// go in, which is expressed by the ORDER the two arms are handed over — so `Pred` has no `Not`.
fn flush_arm(
    body: &mut String,
    pending: &mut Option<(dataflow::ResolvedCond, dataflow::Arm, String, Vec<(usize, String)>)>,
    loops: Vec<(usize, String)>,
    cond: dataflow::ResolvedCond,
    arm: dataflow::Arm,
    mark: String,
) -> Result<(), String> {
    // ⭐ THE SIBLING IS THE SAME PREDICATE'S OTHER SIDE. Anything else means the held arm had no
    // sibling in this walk, so it is emitted one-armed first.
    if let Some((held_cond, held_arm, held_mark, held_loops)) = pending.take() {
        if held_cond == cond && held_arm != arm {
            // ⛔ THE HELD ARM'S OWN LOOPS, not whatever is open now.
            let pred = pred_expr(&held_cond, &held_loops)?;
            let (then_mark, else_mark) = match held_arm {
                dataflow::Arm::Then => (held_mark, mark),
                dataflow::Arm::Else => (mark, held_mark),
            };
            let _ = writeln!(
                body,
                "        e.if_arms(&{pred}, {then_mark}_arm, {else_mark}_arm);"
            );
            return Ok(());
        }
        // ⛔ THE HELD ARM STANDS ALONE. An empty other region prints no `else` at all, which is the
        // one-armed branch — legal, and what the template said.
        emit_one_armed(body, &held_cond, held_arm, &held_mark, &held_loops)?;
    }
    *pending = Some((cond, arm, mark, loops));
    Ok(())
}

/// A BRANCH WITH ONLY ONE SIDE POPULATED.
fn emit_one_armed(
    body: &mut String,
    cond: &dataflow::ResolvedCond,
    arm: dataflow::Arm,
    mark: &str,
    loops: &[(usize, String)],
) -> Result<(), String> {
    let pred = pred_expr(cond, loops)?;
    let (then_arm, else_arm) = match arm {
        dataflow::Arm::Then => (format!("{mark}_arm"), "Arm::empty()".to_owned()),
        dataflow::Arm::Else => ("Arm::empty()".to_owned(), format!("{mark}_arm")),
    };
    let _ = writeln!(body, "        e.if_arms(&{pred}, {then_arm}, {else_arm});");
    Ok(())
}

/// THE PREDICATE, AS A `Pred` EXPRESSION over the ivs of the loops currently open.
///
/// ⛔⛔ `loop_depth` INDEXES THE OPEN LOOPS, NOT THE OPEN CONSTRUCTS. A predicate names an ENCLOSING
/// loop — `constructConditionalOperation` reads `getInductionVar()` of it, so it must be in scope — and
/// the arms interleaved with those loops do not count toward its depth.
///
/// ⛔ A DEPTH WITH NO OPEN LOOP IS A REFUSAL, NOT A GUESS. It would mean a guard over a loop that does
/// not enclose the branch, which has no induction variable to compare against.
fn pred_expr(cond: &dataflow::ResolvedCond, loops: &[(usize, String)]) -> Result<String, String> {
    match cond {
        dataflow::ResolvedCond::Position { loop_stmt, place } => {
            // ⭐⭐ FOUND BY IDENTITY, IN THE SET CAPTURED WHEN THE ARM OPENED. That is the port of
            // `getMLIRLoopFromSNLoopNode(.., dsc_loops_to_mlir_loops_map)` — a lookup keyed by the loop
            // node, resolved against the loops that were actually in scope.
            let mark = loops
                .iter()
                .find_map(|(stmt, mark)| (stmt == loop_stmt).then_some(mark))
                .ok_or_else(|| {
                    format!(
                        "a guard names the loop at statement {loop_stmt}, which is not among the \
                         {} loops enclosing this arm — it has no induction variable in scope \
                         (SNControlFlowLowering.cpp:92-137)",
                        loops.len()
                    )
                })?;
            let place = match place {
                dataflow::Place::First => "First",
                dataflow::Place::Last => "Last",
            };
            Ok(format!(
                "Pred::Position {{ iv: {mark}_iv, place: Place::{place} }}"
            ))
        }
        dataflow::ResolvedCond::All(of) | dataflow::ResolvedCond::Any(of) => {
            let which = if matches!(cond, dataflow::ResolvedCond::All(_)) {
                "All"
            } else {
                "Any"
            };
            let inner: Result<Vec<String>, String> =
                of.iter().map(|one| pred_expr(one, loops)).collect();
            Ok(format!("Pred::{which}(vec![{}])", inner?.join(", ")))
        }
        // ⛔⛔ A NEGATION IS A PREDICATE OPERATION AFTER ALL, and I had it backwards.
        // `SNControlFlowLowering.cpp:181-186` emits `arith.cmpi eq, %composed, %false` and branches the
        // wrapper on THAT; the arms are never swapped — the then region always holds `children[0]`.
        // The op that crashed dbo-opt was `arith.xori`, which is a different op the reference never
        // emits.
        dataflow::ResolvedCond::Not(inner) => Ok(format!(
            "Pred::Not(Box::new({}))",
            pred_expr(inner, loops)?
        )),
    }
}

/// Whether two enclosing constructs are the SAME one, not merely the same shape.
fn same_enclosing(a: &dataflow::Enclosing, b: &dataflow::Enclosing) -> bool {
    match (a, b) {
        (dataflow::Enclosing::Loop { stmt: x }, dataflow::Enclosing::Loop { stmt: y }) => x == y,
        (dataflow::Enclosing::Arm(x), dataflow::Enclosing::Arm(y)) => x == y,
        _ => false,
    }
}
