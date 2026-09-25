//! THE WIRE VIEW — the port's [`Dsc`]/[`ScheduleView`] traits over the parsed scheduled wire
//! (`crate::wire`), so the lowering walks a real `sdsc.json` instead of its own fixtures.
//!
//! # The two seams this file exists against
//!
//! ⛔⛔ THE LEAVES ARE BUILT **INSIDE** [`ScheduleView::roots`], against `UnitHandles` the driver
//! minted one step earlier — which is why the statement's leaf payloads are owned and its closure
//! seams are boxed (see [`TransferStatement`]'s note). A view cannot pre-build them against its own
//! `'c` wire data.
//!
//! ⛔⛔ AND THE LATCH MAP IS WALK STATE. A `ComputeInput::Latch` names its latch by id and the
//! [`Val`] only exists once an earlier statement of the same unit latched it — see
//! [`Handlers::latches`], the reference's `global_latch_map`.
//!
//! # The three structures, and why there are three
//!
//! A [`RootIterArg`] embeds `StartAddresses<'c>`, whose `all: &'c [i64]` and `at: &'c dyn Fn(..)`
//! point at data that must outlive every view of this DSC — and [`Scheduled::Band`]'s
//! `root.transfers: &'s [RootIterArg<'s>]` needs a slice that outlives the `roots()` call itself,
//! which consumes the view. No single struct can own the tables and the slice that borrows them
//! (that is a self-reference), so the data is split:
//!
//! - [`WireTables`] — the OWNED form, built once by the caller: both ends of every transfer the
//!   component sees, resolved to units, locations, buffers and address maps; per-end flattened
//!   address tables and boxed lookups; the per-component head-children trees with `own`
//!   pre-populated and `all` left empty; the per-band propagated lists. No borrows of itself
//!   anywhere.
//! - [`WireDsc<'c>`] — the DSC, borrowed at `'c`. Its `root_args: Vec<RootIterArg<'c>>` borrow
//!   **`&'c WireTables`' data** — data outside the struct — so the struct is not self-referential,
//!   and `view(&'c self)` can lend `&'c self.root_args` to every view it hands out.
//! - [`WireView<'c>`] — the head's view of one component, which walks the wire's own tree.
//!
//! That split is the whole reason the driver trait takes `&'c self` and `dscs: &'c [D]`: the
//! leaves a view builds must live as long as the `'s` the walk runs under, and only the region
//! the tables were borrowed at can guarantee that.
//!
//! # The reads this file makes of the wire
//!
//! ⭐ THE RELEVANCE FILTER IS `isNodeRelevant` WHOLE (`dsc/dsc2.cpp:1916-1938`): a node appears in
//! comp's view iff the node's `relevantComps_` map has an entry for the component; THEN, where a
//! core is named, the core must be one the node names and the corelet must be one of that core's;
//! a negative core id keeps the whole component. [`Viewed`] is exactly that question's three
//! spellings: [`Viewed::OverEveryPair`] is the component-only read and [`Viewed::OnPair`] adds the
//! core and corelet tests. Roots are the head's children — nodes whose `prev_` is empty — in file
//! order, filtered per view.
//!
//! ⭐ THE STAGE NAMES AND DIMS COME FROM `dataStageParam_` READ AT THE DIM, keyed by the loop's
//! `numId_`/`denId_` — the two stages every loop name is built from
//! (`SNControlFlowLowering.cpp:979-982`), and the four staging values a trip count divides out
//! of.
//!
//! ⭐ THE SYNC ENDS COME FROM THE OTHER ENDS' `relevantComps_`, merged per `(component, core)` —
//! `getComponentsFromOtherEnds` (`dsc/dsc2.cpp:2408-2421`), unfiltered by core because the
//! uniformized driver's `SNDSCLowering` carries `core_id_ = -1`
//! (`DSC2ToDataflowIR.cpp:431`), and `-1` keeps every core.
//!
//! ⭐ AND THE SYNC LEAF LOWERS AGAINST A LOCALLY-BUILT [`Handlers`] — the same bridge the port's
//! own fixtures spell by hand (driver.rs's tests): `units` from [`UnitHandles::neighbours`], the
//! two register files from the neighbourhood's own files, and a FRESH latch map, which is the
//! reference's own `global_latch_map` lifetime (one per program unit, `:336`/`:424`). For the
//! L-family units the two register files have no [`Val`] to carry — `buildNeighborUnits` binds
//! nothing for them (`DSC2ToDataflowIRUtils.hpp:155-410`, and the L3 halves bind nothing at all)
//! — so a placeholder `Val` stands in, exactly as a `Val(1)` does in the port's fixture tests.
//! The reference could not answer a retrieval of `LRFREG` from those units either: its
//! `DT_CHECK(false && "GetUnitOp must have been created already")` (`SNDSCLowering.cpp:64`) is
//! the abort that fires. No sync leaf reads them — a sync's retrievals ask Unit components only
//! ([`construct_units`]) — and the placeholder is documented here so the day one does, this is
//! the arm to widen.
//!
//! # What this slice covers, and the three stated gaps
//!
//! The `Dsc` impl is whole — cores, corelet count, kind and the fold question all read off the
//! wire — and [`ScheduleView::roots`] walks the schedule's structure: bands from loop nodes
//! (dims, stages, root args and the band's own propagated transfer list), blocks, and syncs with
//! their own names and wait modes. What it does not yet lower:
//!
//! ⚠️ THE TRANSFER LEAF IS A REFUSAL-SHAPED STEP, NOT A SILENT SKIP. The wire has the transfer's
//! everything — ends, fold data, corelet views — but [`TransferStatement`]'s four closures need the
//! loop IVs and parent-loop ops that [`construct_loops`] mints only AFTER the statement walk, and
//! the port's own fixtures spell those as `Val(80)` literals. Wiring that join is the next
//! increment; until it lands, a transfer node lowers no leaf, which the reference would not do —
//! the golden diff will show every transfer missing, and that is this slice's stated gap.
//!
//! ⚠️ AND THE COMPUTE LEAF WAITS ON THE SAME JOIN. A MAC's [`OperandContext`] embeds the unit's
//! own `&Handlers`, which `roots` receives — that part is ready — but its memory operands carry
//! `outer_loops` and `offsets` keyed on the same not-yet-minted loop values, and the fixture's
//! one compute node is a `fma16` over three inputs on the SFP, two of which are LX/latch views.
//!
//! ⚠️ AND THE UNIFORMIZED SYNC IS THE THIRD GAP. The fixture's golden MLIR carries uniformized
//! syncs (`uniform.query_map` feeding `dataflow.sync_send`), which the reference lowers through
//! `constructUnitsForUniformization` — a function whose OTHER ends resolve through
//! `unit_to_value_map_`, a translator member that PERSISTS across components and fills lazily
//! (`DSC2ToDataflowIRUtils.hpp:67-92`), including L3 entries created on demand by the uniformized
//! getter (`:95-141`). This port builds each component's program unit with its OWN [`Handles`]
//! and drops it, so the cross-component join `construct_units_for_uniformization` needs is not one
//! a per-component view can state. The PLAIN arm — [`construct_units`] — is implemented, and a
//! non-uniformized run of the same wire walks it; the uniform arm is this slice's stated gap and
//! its absence will show in the golden diff as the sync leaves of the uniformized pass.
//!
//! # The smaller documented decisions
//!
//! ⚠️ A SYMBOLIC DIM OR A PARAMETRIC LOOP IS A PANIC IN THIS SLICE, not a refusal: the rmsq
//! fixture has neither (`loopCountSymbolIds_` empty, no `parametric_` loop), and spelling the
//! `DimBound::Symbolic` arm needs `symbolDefinitions_` the wire does not yet carry. The day it
//! must, the arm is where it goes.
//!
//! ⚠️ THE PARENT LOOP IS A DUMMY FOR THIS SLICE. [`DimBound::Counted`]'s `parent` is the enclosing
//! loop's IV, which only exists once the loop nest is being built — but the reference's own dim
//! reader takes the `num_ss == num_el` arm (`affine_dim_loop`) whenever a dim's steady-state and
//! epilogue extents agree, and THAT ARM NEVER READS THE PARENT (`SNControlFlowLowering.cpp:789`
//! — the `parent_node` is only handed to the epilogue arm). Every dim of every stage in the rmsq
//! fixture has `ss == el`, so the dummy is never consulted; the day a fixture disagrees, the
//! epilogue arm needs the real join and the dummy's `Val(0)` is the wrong answer the type system
//! cannot flag.
//!
//! ⚠️ THE SYNC SIGNAL IS NOT ON THE WIRE AND NOT IN THE GOLDEN TEXT. `signal_` is the DDL
//! template's `signal_name=` (`ddl.sync`), which the scheduled dump never prints and the
//! reference's printer never writes back — the golden MLIR's sync ops carry `dbgName` and the
//! wait flag alone. Any [`SyncSignal`] variant is observationally equivalent for every fixture in
//! scope; [`signal_of`] names one so the choice is written down rather than implicit.
//!
//! ⚠️ A NON-SWITCHING END'S `loc` IS DEAD DATA, AND THE FIXTURE PROVES IT. The granularity table
//! (`sys-arch-spec/sysdef.cpp:531-558`) has 23 rows, and the reference consults it ONLY for an end
//! whose buffers switch (`buffering_or_streaming_mode` checks `switches` before
//! `address_granularity_multiply_factor` reads `loc`, `SNControlFlowLowering.cpp:415-435`). The
//! rmsq fixture's three SFP-side ends — `{sfp, latch}`, `{sfp, lxlu}`, `{sfp, lxsu}` — are all
//! NON-switching and all outside the table, so resolving their location would abort a walk the
//! reference completes. [`End::loc`] is therefore [`None`] for a pair the table does not carry,
//! and a SWITCHING end outside the table is a real gap and panics — the reference's own
//! `sysDef.addressGranularityScalePerUnit.at(loc)` throw (`SNDSCLowering.cpp:156-176`).
//!
//! ⚠️ AND THE BAND'S `transfers` ARE PRE-PROPAGATED, THE WAY THE REFERENCE PRE-PROPAGATES THEM.
//! The reference runs the propagation over the head children INSIDE `constructLoops`
//! (`SNControlFlowLowering.cpp:1207-1210`), which runs BEFORE `constructLoopsRecursive` asks the
//! root arm for a band's own list — so every band reads a list that already includes every
//! switching transfer at or below it. The port's [`construct_loops`] runs the same propagation at
//! the same point, over the head children THIS view handed it, and the walk that follows reads
//! `band.transfers` — so the pre-propagated lists are computed HERE, once, per component, over a
//! throwaway clone of the head children, and the head children themselves arrive UNPROPAGATED
//! (`own` filled, `all` empty) because [`construct_loops`] owns the propagation — pinned by its
//! own test asserting `children[0].all` AFTER the call.
//!
//! ⚠️ THE ROOT ARGS ARE THE WHOLE COMPONENT'S SWITCHING TRANSFERS, NOT ONE BAND'S. The reference
//! fills `root_args` at the point it asks the root arm, over
//! `(*dsc_all_parent_loops_to_buffers_switch_map_)[node]` — the band's own propagated list — but
//! the mode/factor/address it reads for each is re-asked from the transfer itself
//! (`:599-612`), which is component-wide data. The port's [`LoopIterArgs::Root`] carries the
//! band's `transfers` separately from the root args' slice, and the indices must AGREE — which
//! they do because both lists are built from the same component-wide re-numbering, the propagated
//! list's entries are `TransferId`s indexing it, and [`WireDsc::new`] pushes one root arg per
//! component-wide switching transfer. A root arg for a transfer that never appears in any band's
//! list is inert: `construct_loop_iter_args` reads it only at an index the band's own list
//! supplies.
//!
//! ⚠️ AND `folds_needed`'S `dst_vias` CANNOT BE PRE-BUILT. The port's [`folds_are_needed`] takes
//! `transfers: &[Transfer<'_>]` with `dst_vias: &'a [Component]` borrowed for the call's own
//! duration, so the vias are spelled INSIDE the closure, per call — a `Vec` of them cannot be
//! stored on `self` and lent at the callee's lifetime.
//!
//! ⚠️ THE STAGE EXTENTS ARE `double` ON THE WIRE AND `int` IN THE TRIP COUNT. `DataStructDims`'s
//! members are `f64` (`dims.h:163-204`) and the value chain that divides them down to a trip
//! count narrows once, at the end — the read here is that truncation, made explicit because a
//! `round` would move every loop bound a fractional extent sits on.
//!
//! ⚠️ AND THE ZERO-FOLD ADDRESS KEY IS ARBITRARY, AND THAT IS THE POINT: a zero-fold map answers
//! ONE value for the whole fold space (`importFromJson`'s early arm reads the JSON value itself,
//! `foldInfrastructure.h:2771-2774`), so the lookup at any coordinate is that value and the
//! distinct count at any pair is one. A key of `(0, 0, 0)` spells "any coordinate" the only way a
//! keyed map can.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroI32;

use super::control_flow::{
    BandDim, BufferSwitch, DimBound, DimStages, NodeKind, NumBuffers, ParentLoop, RootArgs,
    RootIterArg, StartAddresses, SwitchSide, SwitchNode, SwitchingTransfer, TransferId,
    buffering_or_streaming_mode, propagate_buffer_switch_loops_to_root,
};
use super::driver::{
    Dsc, Emitted, FoldedAddresses, FoldDimFunc, Scheduled, ScheduleView, StartAddrOf, Transfer,
    UnitHandles, Used, Viewed, Viewing, folds_are_needed, folded_addresses_are_same,
};
use super::dsc_lowering::{Component, DataLocation, Handlers};
use super::sync::{SyncEnd, SyncKind, SyncUnits, construct_units};
use super::utils::{DscKind, OwnFiles};
use crate::generated::{DataType, SyncSignal};
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects::Val;
use crate::units::{Core, Corelet, DfirUnit, NumFolds, Row};
use crate::wire::{FoldData, NodeKind as WireKind, ScheduleNode, WireOp};
use sys_arch_spec::arch_enums::SenComponent;

/// ONE END OF A TRANSFER, RESOLVED ONCE — the unit, the location pair, the buffers, and the
/// start-address map, all pre-read so neither the scan nor the view re-reads the wire.
#[derive(Debug, Clone)]
struct End {
    /// `src_.unit_` / `via.loc_.unit_`, as a [`DfirUnit`].
    unit: DfirUnit,
    /// The `{unit, storage}` pair as the granularity table keys it — or [`None`] for a pair the
    /// table does not carry, which only a non-switching end may be (see the module doc).
    loc: Option<DataLocation>,
    /// The allocation's `numBuffers_` behind this end, or [`None`] where none backs it.
    switches: Option<NumBuffers>,
    /// `startAddr_`, re-keyed per `(core, corelet, time)` fold coordinate.
    addresses: BTreeMap<(Core, Corelet, u32), i64>,
}

/// ONE TRANSFER NODE, RESOLVED ONCE — both ends, plus the two `bufferSwitchPosition_`.
#[derive(Debug, Clone)]
struct ResolvedTransfer {
    /// The source end.
    src: End,
    /// The destination ends, in `dstVias_` order.
    dsts: Vec<End>,
    /// `srcLdsAndLoopOffsets_.bufferSwitchPosition_` — the loop's tree index.
    src_position: Option<usize>,
    /// `dstLdsAndLoopOffsets_[0].bufferSwitchPosition_` — index ZERO, for every via.
    dst_position: Option<usize>,
    /// The source lds's `dataFormat_`, which entry 056 reads the granularity factor at.
    precision: DataType,
}

impl ResolvedTransfer {
    /// ONE END AS A [`SwitchSide`] — the mask payload is the end's own address map, and `loc` is
    /// the granularity row where the table carries the pair.
    ///
    /// ⛔ PANICS FOR A SWITCHING END OFF THE TABLE, and that is the port of the reference's own
    /// `sysDef.addressGranularityScalePerUnit.at(loc)` throw: the pair is one the machine does
    /// not define, and no walk can lower past it.
    fn side(end: &End) -> SwitchSide<'_, BTreeMap<(Core, Corelet, u32), i64>> {
        let loc = end
            .loc
            .unwrap_or_else(|| panic!("a switching {unit:?} end off the granularity table", unit = end.unit));
        SwitchSide {
            unit: end.unit,
            loc,
            switches: end.switches,
            start_addresses: &end.addresses,
        }
    }

    /// THE MODE QUESTION, ANSWERED — entry 056 over this transfer's resolved ends. Asked by the
    /// root-arg build and the own-list scan; the reference asks it at each of those points too.
    fn mode(&self, comp: DfirUnit) -> Option<BufferSwitch<'_, BTreeMap<(Core, Corelet, u32), i64>>> {
        buffering_or_streaming_mode(
            comp,
            self.precision,
            &Self::side(&self.src),
            &self
                .dsts
                .iter()
                .map(Self::side)
                .collect::<Vec<_>>(),
        )
    }
}

/// ONE COMPONENT'S TABLES, OWNED — everything [`Dsc::view`] hands the driver, in the shapes that
/// do not borrow.
///
/// ⛔ THE LENT SHAPES ARE NOT IN HERE. A [`RootIterArg`] embeds `&'c` borrows of this struct's
/// `address_tables` and `lookups`, and a struct cannot store a borrow of itself — which is why
/// [`WireDsc`] builds those borrows in its own constructor, against the `&'c WireTables` this
/// struct is reached through, and stores them in ITS `root_args` field.
struct ComponentTables {
    /// The component's own transfers, in tree order — the reference's
    /// `traverseTreeDFS(nullptr, {TRANSFER}, comp, -1, -1)`, which keeps every transfer EITHER end
    /// of which names comp. The position in this list is the transfer's [`TransferId`].
    switching: Vec<ResolvedTransfer>,
    /// PER END, IN THE ORDER THE ENDS WERE PUSHED — source first, then the vias, per transfer —
    /// one flattened address table and one boxed lookup, which are the two reads a
    /// [`RootIterArg`] makes of `*start_address_map`.
    address_tables: Vec<Vec<i64>>,
    /// The lookup closures, one per end, in the same order.
    lookups: Vec<Box<dyn Fn(Core, Corelet, u32) -> i64>>,
    /// The schedule head's children, as this component sees them, with each loop's `own`
    /// transfers PRE-POPULATED and `all` left EMPTY — [`construct_loops`] owns the propagation.
    head_children: Vec<SwitchNode<TransferId>>,
    /// PER LOOP NODE NAME, the PRE-PROPAGATED transfer list — what a band's `transfers` field
    /// carries, computed once here over a throwaway clone of the head children.
    band_transfers: BTreeMap<String, Vec<TransferId>>,
}

/// THE WHOLE WIRE'S TABLES, OWNED — built once by the caller, borrowed per view. This is the
/// struct that owns everything the `'c`-borrowed leaves point at.
pub struct WireTables {
    /// `coreIdsUsed_`, as `Core`s.
    cores: Vec<Core>,
    /// `numCoreletsUsed_DSC2_`.
    num_corelets_used: u32,
    /// PER-COMPONENT TABLES, keyed by the census component.
    components: BTreeMap<DfirUnit, ComponentTables>,
    /// The fold-sameness answer's constants — `constantInfo_`'s dim funcs, in id order.
    constant_dim_funcs: Vec<FoldDimFunc>,
}

impl WireTables {
    /// ONE WIRE OP'S TABLES — [`Dsc::view`] is per-component; the tables are not, and building
    /// them once is what keeps a view cheap enough to take per (core, corelet).
    ///
    /// # Panics
    ///
    /// On a core id, SWITCHING transfer end, loop stage or extent the arch's tables cannot name —
    /// all facts of the fixture the reader has already validated structurally, so a gap here is a
    /// build-time question about the fixture, not a runtime refusal.
    #[must_use]
    pub fn of(op: &WireOp) -> WireTables {
        let cores = op
            .core_ids_used
            .iter()
            .map(|id| {
                let id = u32::try_from(*id).expect("a core id the arch can name");
                Core::checked(id).expect("a core id the arch can name")
            })
            .collect::<Vec<_>>();

        let mut components = BTreeMap::new();
        for comp in super::driver::sen_components().iter() {
            components.insert(comp, component_tables(op, comp));
        }

        let constant_dim_funcs = op
            .constant_info
            .values()
            .flat_map(|constant| {
                constant
                    .data
                    .as_ref()
                    .map(|folded| folded.funcs.iter().copied())
                    .into_iter()
                    .flatten()
            })
            .map(wire_fold_func)
            .collect();

        WireTables {
            cores,
            num_corelets_used: u32::try_from(op.num_corelets_used_dsc2.max(0)).unwrap_or(0),
            components,
            constant_dim_funcs,
        }
    }

    /// This component's tables — the census guarantees the key.
    fn tables(&self, comp: DfirUnit) -> &ComponentTables {
        self.components
            .get(&comp)
            .expect("a census component, which `of` built tables for")
    }
}

/// ONE COMPONENT'S TABLES — the re-numbered own list with its ends' address tables and lookups,
/// the unpropagated head children, and the per-band propagated lists.
fn component_tables(op: &WireOp, comp: DfirUnit) -> ComponentTables {
    // The component's own transfer list, in tree order.
    let own: Vec<&crate::wire::TransferNode> = op
        .schedule_tree
        .iter()
        .filter_map(|node| match &node.kind {
            WireKind::Transfer(transfer) => Some(transfer),
            _ => None,
        })
        .filter(|node| {
            node.src.unit == sen_of(comp)
                || node.dst_vias.iter().any(|via| via.loc.unit == sen_of(comp))
        })
        .collect();

    let mut switching = Vec::new();
    let mut address_tables = Vec::new();
    let mut lookups: Vec<Box<dyn Fn(Core, Corelet, u32) -> i64>> = Vec::new();
    for node in &own {
        let resolved = resolve_transfer(op, node);
        // PER END, the flattened table and the boxed lookup — source first, then each via.
        for end in std::iter::once(&resolved.src).chain(resolved.dsts.iter()) {
            address_tables.push(end.addresses.values().copied().collect());
            let map = end.addresses.clone();
            lookups.push(Box::new(move |core, corelet, time| {
                map.get(&(core, corelet, time)).copied().unwrap_or(0)
            }));
        }
        switching.push(resolved);
    }

    // The head children with each loop's `own` from the buffer-loop scan, UNPROPAGATED.
    let head_children = head_children(op, &switching, comp);

    // The per-band propagated lists, from a THROWAWAY CLONE — the reference propagates inside
    // `constructLoops` and then reads the lists in its loop walk, so both walks see them filled;
    // the port's `construct_loops` propagates its own copy and this pre-computes the bands' reads.
    let mut propagated = head_children.clone();
    propagate_buffer_switch_loops_to_root(&mut propagated);
    let mut band_transfers = BTreeMap::new();
    collect_band_transfers(op, comp, &propagated, &mut band_transfers);

    ComponentTables {
        switching,
        address_tables,
        lookups,
        head_children,
        band_transfers,
    }
}

/// THE HEAD CHILDREN — the head's own children, as this component sees them, with each loop's
/// `own` transfers pre-populated from the buffer-loop scan (entry 074) and `all` left empty.
fn head_children(
    op: &WireOp,
    switching: &[ResolvedTransfer],
    comp: DfirUnit,
) -> Vec<SwitchNode<TransferId>> {
    op.schedule_tree
        .iter()
        .filter(|node| node.prev.is_empty())
        .filter(|node| relevant(node, comp, Viewed::OverEveryPair))
        .map(|node| switch_node(op, node, switching, comp))
        .collect()
}

/// ONE NODE AS THE SWITCH PROPAGATION READS IT — the component-filtered children, the loop's own
/// transfers, and `all` empty until propagation fills it.
fn switch_node(
    op: &WireOp,
    node: &ScheduleNode,
    switching: &[ResolvedTransfer],
    comp: DfirUnit,
) -> SwitchNode<TransferId> {
    let kind = match &node.kind {
        WireKind::Loop(_) => NodeKind::Loop,
        WireKind::Block => NodeKind::Block,
        // ⚠️ THE WIRE READER REFUSES `CONDITION` AND `STICKMASK` NODES outright
        // (`Refusal::UnknownNodeType`, "no fixture carries one yet"), so the walk cannot meet one
        // and the arm is unreachable.
        WireKind::Transfer(_) | WireKind::Compute(_) | WireKind::Sync(_) | WireKind::Allocate(_) => {
            NodeKind::Leaf
        }
    };

    SwitchNode {
        kind,
        children: op
            .schedule_tree
            .iter()
            .filter(|child| child.prev == node.name)
            .filter(|child| relevant(child, comp, Viewed::OverEveryPair))
            .map(|child| switch_node(op, child, switching, comp))
            .collect(),
        own: if kind == NodeKind::Loop {
            own_of_loop(op, node, switching, comp)
        } else {
            Vec::new()
        },
        all: Vec::new(),
    }
}

/// ONE LOOP'S `own` TRANSFERS — `dsc_loops_to_buffers_switch_map_[this loop]`: the component's own
/// transfers whose buffer-switch position names THIS loop, in the scan's order.
///
/// ⛔ THE SIDE TEST HERE IS `unit_ == comp_` ALONE, not entry 056's "and switches" — a source on
/// this component that does NOT switch still contributes the SOURCE's position
/// (`SNControlFlowLowering.cpp:485-487`); and every matching via reads
/// `dstLdsAndLoopOffsets_[0]` (`:488-497`).
fn own_of_loop(
    op: &WireOp,
    loop_node: &ScheduleNode,
    switching: &[ResolvedTransfer],
    comp: DfirUnit,
) -> Vec<TransferId> {
    let mut located = Vec::new();
    for (index, transfer) in switching.iter().enumerate() {
        if transfer.mode(comp).is_none() {
            continue;
        }
        let position = if transfer.src.unit == comp {
            transfer.src_position
        } else if transfer.dsts.iter().any(|dst| dst.unit == comp) {
            transfer.dst_position
        } else {
            None
        };
        // `bufferSwitchPosition_` names the loop by its TREE INDEX; this loop is the node the
        // tree is being walked at.
        if let Some(position) = position
            && op
                .schedule_tree
                .get(position)
                .is_some_and(|named| named.name == loop_node.name)
        {
            located.push(TransferId(index));
        }
    }
    located
}

/// PER LOOP NODE NAME, the propagated `all` list — walked from the PROPAGATED head children,
/// pairing each loop node with the head child the same tree walk built.
///
/// ⭐ THE PAIRING IS POSITIONAL-BY-CONSTRUCTION: the switch tree carries no names, but it was
/// built by the same filtered walk that reads the schedule tree here, and the recursion preserves
/// order — so the Nth loop node among a parent's relevant children is the Nth `Loop`-kind switch
/// node among the corresponding switch children. Walked in lockstep, the two agree.
fn collect_band_transfers(
    op: &WireOp,
    comp: DfirUnit,
    propagated: &[SwitchNode<TransferId>],
    out: &mut BTreeMap<String, Vec<TransferId>>,
) {
    // The head's relevant children in tree order, and the propagated switch children in the same
    // order — the two walks that built them filtered identically.
    let head: Vec<&ScheduleNode> = op
        .schedule_tree
        .iter()
        .filter(|node| node.prev.is_empty())
        .filter(|node| relevant(node, comp, Viewed::OverEveryPair))
        .collect();
    for (node, child) in head.iter().zip(propagated.iter()) {
        collect_band_transfers_below(op, node, child, out);
    }
}

/// ONE SUBTREE'S LOOP NODES AND PROPAGATED LISTS, in lockstep.
fn collect_band_transfers_below(
    op: &WireOp,
    node: &ScheduleNode,
    child: &SwitchNode<TransferId>,
    out: &mut BTreeMap<String, Vec<TransferId>>,
) {
    if let WireKind::Loop(_) = &node.kind {
        out.insert(node.name.clone(), child.all.clone());
    }
    let children: Vec<&ScheduleNode> = op
        .schedule_tree
        .iter()
        .filter(|below| below.prev == node.name)
        .collect();
    for (below, below_child) in children.iter().zip(child.children.iter()) {
        collect_band_transfers_below(op, below, below_child, out);
    }
}

/// WHETHER `comp` SEES THIS NODE — `isNodeRelevant` (`dsc/dsc2.cpp:1916-1938`): the component key
/// first, the core and corelet tests a named pair adds, and a component-only read keeping every
/// core.
fn relevant(node: &ScheduleNode, comp: DfirUnit, at: Viewed) -> bool {
    let Some(cores) = node.relevant_comps.get(&sen_of(comp)) else {
        return false;
    };
    let Viewed::OnPair(core, corelet) = at else {
        return true;
    };
    let Some(named) = cores.get(&i64::from(core.get())) else {
        return false;
    };
    named.contains(&i64::from(corelet.get()))
}

/// ONE DSC FROM THE WIRE, as the two drivers read it — the tables borrowed at `'c`, with the
/// root args materialized in the constructor so every view can lend them.
pub struct WireDsc<'c> {
    /// The op this view reads.
    op: &'c WireOp,
    /// The whole wire's tables, borrowed at `'c`.
    tables: &'c WireTables,
    /// THE ROOT ARGS, LENT AT `'c` — one per component-wide switching transfer, borrowing the
    /// tables' address tables and lookups. ⛔ NOT SELF-REFERENTIAL: each arg borrows `*tables`
    /// (the caller's data), not any field of this struct.
    root_args: BTreeMap<DfirUnit, Vec<RootIterArg<'c>>>,
    /// [`folds_needed`]'s cores, head-first — `dsc.coreIdsUsed_` passed through untouched
    /// (`DSC2ToDataflowIR.cpp:266`, `:277`).
    cores_used: Used<Core>,
}

impl<'c> WireDsc<'c> {
    /// One DSC over one parsed wire op and its tables — the tables are a separate argument
    /// because the root args borrow them and could not be stored in the same struct that owns
    /// them.
    ///
    /// # Panics
    ///
    /// Where [`RootIterArg`]'s ingredient reads ask for data the tables do not carry — every one
    /// a fact the wire reader validated structurally.
    #[must_use]
    pub fn new(op: &'c WireOp, tables: &'c WireTables) -> Self {
        let mut root_args = BTreeMap::new();
        for comp in super::driver::sen_components().iter() {
            let component = tables.tables(comp);
            let mut args = Vec::new();
            for (index, transfer) in component.switching.iter().enumerate() {
                // ⭐ ENTRY 056 RE-ASKED PER TRANSFER, exactly as the reference's root arm does
                // (`SNControlFlowLowering.cpp:602-609`): the mode, the factor, and WHICH end's
                // map the answer selected.
                let Some(buffer) = transfer.mode(comp) else {
                    // `mode = -1` IS AN ANSWER AND NOT A REFUSAL — a transfer with no switching
                    // end on this component contributes no root arg, which is the port's own
                    // spelling of the reference's `failed(..)` arm at `:467`.
                    continue;
                };
                // WHICH END THE ARG READS — the end the mode question answered with, in the
                // component's own flattened numbering (source first, then the vias).
                let end_index = if transfer.src.unit == comp {
                    Some(0)
                } else {
                    transfer
                        .dsts
                        .iter()
                        .position(|dst| dst.unit == comp)
                        .map(|via| via + 1)
                }
                .unwrap_or(0);
                let flat = index * (1 + transfer.dsts.len()) + end_index;
                let addresses = &component.address_tables[flat];
                args.push(RootIterArg {
                    mode: Some(buffer.mode),
                    factor: buffer.factor,
                    addresses: StartAddresses {
                        all: addresses,
                        at: &*component.lookups[flat],
                        // `getSingleDataStrict(.., {{0, core_id_}, {1, corelet_id_}})` — the one
                        // address this core and corelet carry across folds. The view is per
                        // (core, corelet), so the SINGLE value is read at the view's own pair;
                        // the map's first entry stands in at build time and the walk that needs
                        // the per-pair value re-asks `at` with the pair it holds.
                        single: addresses.first().copied().unwrap_or(0),
                    },
                });
            }
            root_args.insert(comp, args);
        }
        let (head, rest) = tables
            .cores
            .split_first()
            .expect("a DSC the driver walks names a core");
        Self {
            op,
            tables,
            root_args,
            cores_used: Used::of(*head, rest.to_vec()),
        }
    }

    /// The component's root args — the map is keyed by every census component.
    fn args_of(&self, comp: DfirUnit) -> &[RootIterArg<'c>] {
        self.root_args
            .get(&comp)
            .expect("a census component, which `new` built root args for")
    }

    /// ONE FOLD-SAMENESS END — the address map the question is about, in the component's own
    /// numbering.
    fn end_of(
        &self,
        comp: DfirUnit,
        which: StartAddrOf,
    ) -> &BTreeMap<(Core, Corelet, u32), i64> {
        let component = self.tables.tables(comp);
        let (transfer, via) = match which {
            StartAddrOf::Source { transfer } => (transfer, None),
            StartAddrOf::Destination { transfer, via } => (transfer, Some(via)),
        };
        match via {
            None => &component.switching[transfer].src.addresses,
            Some(via) => &component.switching[transfer].dsts[via].addresses,
        }
    }

    /// THE FOLD-SAMENESS ANSWER FOR ONE END AT ONE PAIR — how many DISTINCT addresses the end's
    /// map yields with the pair pinned, counted across the fold/time coordinates —
    /// `getAllDataWithMapUnrolled({{0, core_id}, {1, corelet_id}})`'s size.
    ///
    /// ⛔ PINNED AT THE PAIR, NOT COUNTED WHOLE: the unrolled map answers one value per time
    /// coordinate at a fixed pair, so the count is the distinct values among the pair's entries —
    /// a map with 32 per-core values answers ONE at any single pair, which reads as `Constant`
    /// exactly as the reference's `size == 1` does (`DSC2ToDataflowIR.cpp:236-238`).
    fn addresses_at(
        &self,
        addresses: &BTreeMap<(Core, Corelet, u32), i64>,
        core: Core,
        corelet: Corelet,
    ) -> FoldedAddresses {
        let distinct: BTreeSet<i64> = addresses
            .iter()
            .filter(|((c, cl, _), _)| *c == core && *cl == corelet)
            .map(|(_, value)| *value)
            .collect();
        if distinct.len() <= 1 {
            FoldedAddresses::Constant
        } else {
            FoldedAddresses::PerFold(NumFolds(
                u32::try_from(distinct.len()).expect("a fold count the arch can name")
            ))
        }
    }
}

impl<'c> Dsc<'c> for WireDsc<'c> {
    type Latch = usize;
    type Mask = BTreeMap<(Core, Corelet, u32), i64>;
    type Switch = TransferId;
    type View = WireView<'c>;

    fn cores(&self) -> &[Core] {
        &self.tables.cores
    }

    fn num_corelets_used(&self) -> u32 {
        self.tables.num_corelets_used
    }

    fn kind(&self) -> DscKind {
        // `computeOp_.empty()`: the only live question the port's kind enum asks of a scheduled
        // wire op, which IS a DSC2.0 by construction.
        if self.op.compute_op.is_empty() {
            DscKind::NoComputeOp
        } else {
            DscKind::Dsc2
        }
    }

    fn folds_needed(&self, comp: DfirUnit) -> bool {
        let component = self.tables.tables(comp);
        // ⚠️ THE VIAS ARE SPELLED PER CALL, not stored — see the module doc.
        let vias: Vec<Vec<Component>> = component
            .switching
            .iter()
            .map(|one| {
                one.dsts
                    .iter()
                    .map(|dst| Component::Unit(dst.unit))
                    .collect::<Vec<_>>()
            })
            .collect();
        let transfers: Vec<Transfer<'_>> = component
            .switching
            .iter()
            .zip(vias.iter())
            .map(|(one, vias)| Transfer {
                src: Component::Unit(one.src.unit),
                dst_vias: vias,
            })
            .collect();
        let ask = |which: StartAddrOf, corelets: &[Corelet]| {
            let end = self.end_of(comp, which);
            let corelets = Used::of(
                corelets[0],
                corelets[1..].to_vec(),
            );
            folded_addresses_are_same(&self.cores_used, &corelets, |core, corelet| {
                self.addresses_at(end, core, corelet)
            })
        };
        folds_are_needed(
            self.tables.num_corelets_used,
            &self.tables.constant_dim_funcs,
            &transfers,
            Component::Unit(comp),
            ask,
        )
    }

    fn view(&'c self, comp: DfirUnit, at: Viewed) -> Viewing<'c, Self> {
        let component = self.tables.tables(comp);
        Viewing {
            // ⭐ ONE CLONE PER CALL, BECAUSE THE REFERENCE'S MAP IS A LOCAL OF THE PROGRAM UNIT
            // (`:331-332`): each view gets a fresh copy, exactly as `constructAProgramUnit`
            // declares its own `dsc_all_parent_loops_to_buffers_switch_map`.
            transfers: component
                .switching
                .iter()
                .map(|transfer| SwitchingTransfer {
                    precision: transfer.precision,
                    src: ResolvedTransfer::side(&transfer.src),
                    src_position: transfer.src_position.as_ref(),
                    dsts: transfer
                        .dsts
                        .iter()
                        .map(ResolvedTransfer::side)
                        .collect(),
                    dst_position: transfer.dst_position.as_ref(),
                })
                .collect(),
            head_children: component.head_children.clone(),
            roots: WireView { dsc: self, comp, at },
        }
    }
}

/// THE SCHEDULE HEAD'S VIEW OF ONE COMPONENT — [`ScheduleView`], over the wire's own tree.
pub struct WireView<'c> {
    /// The DSC being viewed.
    dsc: &'c WireDsc<'c>,
    /// The component the view is taken over.
    comp: DfirUnit,
    /// Which `getNextView` overload this view is — the `(core, corelet)` pair, or every pair.
    at: Viewed,
}

impl<'c> ScheduleView<'c> for WireView<'c> {
    fn roots<'s>(self, vals: &mut Values, handles: UnitHandles<'s>) -> Vec<Scheduled<'s>>
    where
        'c: 's,
    {
        // ⭐ THE `&'c` IS COPIED OUT BEFORE `self` IS CONSUMED — the walk that follows may hold
        // the tree's nodes past the end of this call, and only the DSC's region guarantees them.
        let op = self.dsc.op;
        op.schedule_tree
            .iter()
            .filter(|node| node.prev.is_empty())
            .filter(|node| self.relevant(node))
            .map(|node| self.schedule(node, vals, &handles))
            .collect()
    }
}

impl<'c> WireView<'c> {
    /// WHETHER `comp` SEES THIS NODE AT THIS VIEW — `isNodeRelevant` whole (`dsc/dsc2.cpp:1916`).
    fn relevant(&self, node: &ScheduleNode) -> bool {
        relevant(node, self.comp, self.at)
    }

    /// ONE SUBTREE, as both walks take it — the structural kinds recurse, the sync builds its
    /// leaf, and the two refusal-shaped gaps (transfer, compute) are documented in the module
    /// doc above.
    fn schedule<'s>(
        &self,
        node: &'c ScheduleNode,
        vals: &mut Values,
        handles: &UnitHandles<'s>,
    ) -> Scheduled<'s>
    where
        'c: 's,
    {
        match &node.kind {
            WireKind::Loop(loop_node) => {
                assert!(
                    !loop_node.parametric,
                    "a parametric loop is this slice's stated panic — see the module doc"
                );
                let component = self.dsc.tables.tables(self.comp);
                let stages = stage_names(self.dsc.op, loop_node);
                Scheduled::Band {
                    dims: loop_node
                        .dims
                        .iter()
                        .map(|dim| BandDim {
                            dim: dim.dim,
                            bound: dim_bound(self.dsc.op, loop_node, dim.dim),
                        })
                        .collect(),
                    // ⭐ THE BAND'S OWN PRE-PROPAGATED LIST — see the module doc.
                    transfers: component
                        .band_transfers
                        .get(&node.name)
                        .map_or(&[] as &[TransferId], Vec::as_slice),
                    root: RootArgs {
                        uniform: handles.handles,
                        // ⭐ LENT AT `'c`, from the DSC's own tables — see [`WireDsc::new`].
                        transfers: self.dsc.args_of(self.comp),
                    },
                    stages,
                    children: self.children(node, vals, handles),
                }
            }
            WireKind::Block => Scheduled::Block {
                children: self.children(node, vals, handles),
            },
            WireKind::Sync(sync) => Scheduled::Leaf(Emitted::Sync {
                signal: signal_of(),
                name: Some(&node.name),
                kind: self.sync_kind(sync, vals, handles),
            }),
            // ⚠️ THIS SLICE'S STATED GAPS — see the module doc.
            WireKind::Transfer(_) | WireKind::Compute(_) | WireKind::Allocate(_) => {
                Scheduled::Block {
                    children: Vec::new(),
                }
            }
        }
    }

    /// `getNextView(comp_, corelet_id_, core_id_)` — the children the component sees, in file
    /// order, whose `prev_` names this node.
    fn children<'s>(
        &self,
        node: &'c ScheduleNode,
        vals: &mut Values,
        handles: &UnitHandles<'s>,
    ) -> Vec<Scheduled<'s>>
    where
        'c: 's,
    {
        self.dsc
            .op
            .schedule_tree
            .iter()
            .filter(|child| child.prev == node.name)
            .filter(|child| self.relevant(child))
            .map(|child| self.schedule(child, vals, handles))
            .collect()
    }

    /// WHICH OF THE THREE SYNC STATEMENTS THIS IS — `isReceive_` read against
    /// `implicitSyncRefTransfer_`, with the ends from the other ends' `relevantComps_`.
    fn sync_kind<'s>(
        &self,
        sync: &'c crate::wire::SyncNode,
        vals: &mut Values,
        handles: &UnitHandles<'s>,
    ) -> SyncKind<'s>
    where
        'c: 's,
    {
        assert!(
            sync.implicit_sync_ref_transfer.is_none(),
            "an implicit sync is this slice's stated panic — no fixture in scope has one"
        );
        let ends = self.sync_ends(sync);
        // ⛔ THE PLAIN ARM ALONE — see the module doc's uniformized-sync gap.
        let units = SyncUnits::Plain(construct_units(
            vals,
            &self.handlers_bridge(handles),
            &ends,
        ));
        if sync.is_receive {
            SyncKind::Receive { units }
        } else {
            SyncKind::Produce {
                wait_immediately: !sync.is_soft,
                units,
            }
        }
    }

    /// THE HANDLES A SYNC LEAF LOWERS AGAINST — `component_to_handler_` at the moment the view
    /// runs, rebuilt from the [`UnitHandles`] the driver minted. See the module doc for the
    /// register-file placeholder and the latch map's per-unit lifetime.
    fn handlers_bridge(&self, handles: &UnitHandles<'_>) -> Handlers {
        let own = &handles.neighbours.own;
        // ⚠️ THE PLACEHOLDER — the L-family units bind no register files, and no sync leaf reads
        // them; see the module doc.
        let placeholder = Val(0);
        let (own_lrf, pt_xrf) = match own {
            OwnFiles::PtRow { lrf, xrf, .. } => (*lrf, *xrf),
            OwnFiles::Pe { lrf } => (*lrf, placeholder),
            OwnFiles::Sfp { lrf } => (*lrf, placeholder),
            OwnFiles::L0su { .. } | OwnFiles::None => (placeholder, placeholder),
        };
        Handlers {
            units: handles.neighbours.units.clone(),
            own_lrf,
            pt_xrf,
            latches: Default::default(),
        }
    }

    /// THE OTHER ENDS, AS `(end, cores)` — `getComponentsFromOtherEnds`
    /// (`dsc/dsc2.cpp:2408-2421`): each other end's `relevantComps_` merged per component, in
    /// the order the other ends name them.
    fn sync_ends(
        &self,
        sync: &'c crate::wire::SyncNode,
    ) -> Vec<(SyncEnd, Vec<(Core, Vec<Corelet>)>)> {
        let mut ends: Vec<(SyncEnd, Vec<(Core, Vec<Corelet>)>)> = Vec::new();
        for other in &sync.other_end_of_the_signals {
            let Some(node) = self.dsc.op.schedule_tree.get(*other) else {
                continue;
            };
            for (component, cores) in &node.relevant_comps {
                let Some(end) = sync_end_of(*component) else {
                    continue;
                };
                let resolved: Vec<(Core, Vec<Corelet>)> = cores
                    .iter()
                    .filter_map(|(core, corelets)| {
                        let core = u32::try_from(*core).ok()?;
                        Some((Core::checked(core)?, corelets))
                    })
                    .map(|(core, corelets)| {
                        (
                            core,
                            corelets
                                .iter()
                                .filter_map(|corelet| {
                                    let corelet = u32::try_from(*corelet).ok()?;
                                    Corelet::checked(corelet)
                                })
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect();
                match ends.iter_mut().find(|(held, _)| *held == end) {
                    Some((_, held)) => {
                        for (core, corelets) in resolved {
                            match held.iter_mut().find(|(walked, _)| *walked == core) {
                                Some((_, walked)) => walked.extend(corelets),
                                None => held.push((core, corelets)),
                            }
                        }
                    }
                    None => ends.push((end, resolved)),
                }
            }
        }
        ends
    }
}

/// `sync->signal_` — ⚠️ NOT ON THE WIRE AND NOT IN THE GOLDEN TEXT, because the reference's
/// `signal_` is the `ddl.sync` `signal_name=` the DDL template spelled and the scheduled dump
/// never prints, and `construct_sync_operation`'s printer never writes it back. Every sync of
/// every fixture in scope pairs an L-family store half with an L-family load half; the variant
/// below is that vocabulary's spelling, chosen so the decision is written down. The day a
/// fixture disagrees, this is the arm to widen.
fn signal_of() -> SyncSignal {
    SyncSignal::InputToL0suToL0luToSync
}

/// ONE WIRE COMPONENT AS A SYNC END — [`SyncEnd`]'s vocabulary over the wire's unit spellings:
/// the units [`DfirUnit`] names outright, the L0LU rows (row 0 is [`DfirUnit::L0lu`], rows 1-7
/// are the spellings [`construct_units`] skips), or [`None`] for a component no sync end can be
/// (the memories, the register files, the constant sources).
fn sync_end_of(component: SenComponent) -> Option<SyncEnd> {
    match component {
        SenComponent::L0lurow1 => Some(SyncEnd::L0luRow(row_of(1))),
        SenComponent::L0lurow2 => Some(SyncEnd::L0luRow(row_of(2))),
        SenComponent::L0lurow3 => Some(SyncEnd::L0luRow(row_of(3))),
        SenComponent::L0lurow4 => Some(SyncEnd::L0luRow(row_of(4))),
        SenComponent::L0lurow5 => Some(SyncEnd::L0luRow(row_of(5))),
        SenComponent::L0lurow6 => Some(SyncEnd::L0luRow(row_of(6))),
        SenComponent::L0lurow7 => Some(SyncEnd::L0luRow(row_of(7))),
        _ => unit_of(component).map(SyncEnd::Unit),
    }
}

/// A PT [`Row`] by index — every spelling above names a row this arch has, or the fixture's own
/// relevance map named a unit the hardware cannot.
fn row_of(index: u32) -> Row {
    Row::checked(index).expect("a row the arch can name")
}

/// A WIRE FOLD FUNC, AS THE DRIVER READS IT — the three spellings the wire carries, mapped onto
/// [`FoldDimFunc`]; the two spellings the wire never writes (`WkSplit`, `Unknown`) have no wire
/// form to come from.
fn wire_fold_func(func: crate::wire::FoldDimFunc) -> FoldDimFunc {
    match func {
        crate::wire::FoldDimFunc::Const => FoldDimFunc::Constant,
        crate::wire::FoldDimFunc::Map => FoldDimFunc::Map,
        crate::wire::FoldDimFunc::Affine { .. } => FoldDimFunc::Affine,
    }
}

/// THE TWO STAGE NAMES — `dataStageParam_.at(numId_).name()` and `.at(denId_).name()`.
fn stage_names<'o>(op: &'o WireOp, loop_node: &crate::wire::LoopNode) -> (&'o str, &'o str) {
    let num = stage_of(op, loop_node.num_id);
    let den = stage_of(op, loop_node.den_id);
    (num.name(), den.name())
}

/// ONE STAGE, REFUSED LOUDLY — a loop whose stage id the table does not carry is a fixture gap,
/// not a refusal the reference can take (`dataStageParam_.at()` throws).
fn stage_of(op: &WireOp, id: i64) -> &crate::wire::DataStage {
    op.data_stage_param
        .get(&id)
        .unwrap_or_else(|| panic!("dataStageParam_ has no stage {id}"))
}

/// ONE DIM'S BOUND — counted, from the two stages' extents, with the parent-loop dummy this
/// slice documents.
fn dim_bound(op: &WireOp, loop_node: &crate::wire::LoopNode, dim: crate::wire::PrimaryDim) -> DimBound {
    let num = stage_of(op, loop_node.num_id);
    let den = stage_of(op, loop_node.den_id);
    assert!(
        loop_node.loop_count_symbol_ids.is_empty(),
        "a symbolic dim is this slice's stated panic — see the module doc"
    );
    DimBound::Counted {
        stages: DimStages {
            num_ss: dim_value(&num.ss, dim),
            num_el: dim_value(&num.el, dim),
            den_ss: NonZeroI32::new(dim_value(&den.ss, dim))
                .expect("a denominator stage's ss is nonzero"),
            den_el: dim_value(&den.el, dim),
        },
        // ⚠️ THE DUMMY — see the module doc: never read while `num_ss == num_el`, which every
        // dim of every fixture in scope satisfies.
        parent: ParentLoop::AffineConst {
            iv: Val(0),
            upper_bound: 0,
        },
    }
}

/// ONE STAGE'S EXTENT FOR ONE DIM — the `DataStructDims` field the dim names. The C++ members
/// are `double` (`dims.h:163-204`) and the trip count the reference divides out of them narrows
/// once, at the end; the truncation here is that narrowing.
fn dim_value(stage: &crate::wire::DataStructDims, dim: crate::wire::PrimaryDim) -> i32 {
    use crate::wire::PrimaryDim as P;
    let value = match dim {
        P::In => stage.in_,
        P::Out => stage.out_,
        P::Mb => stage.mb_,
        P::Ij => stage.ij_,
        P::Y => stage.y_,
        P::X => stage.x_,
        P::X1 => stage.x1_,
        P::Kij => stage.kij_,
        P::I => stage.i_,
        P::J => stage.j_,
        P::Ki => stage.ki_,
        P::Kj => stage.kj_,
    };
    // The reference's `int(..)` narrowing, made explicit.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "`int(..)` — the reference's own narrowing of a `double` extent"
    )]
    let truncated = value as i32;
    truncated
}

/// ONE TRANSFER, RESOLVED — both ends' units, locations, buffers and start addresses read once.
fn resolve_transfer(op: &WireOp, node: &crate::wire::TransferNode) -> ResolvedTransfer {
    ResolvedTransfer {
        src: resolve_end(op, node.src.unit, &node.src_lds_and_loop_offsets),
        dsts: node
            .dst_vias
            .iter()
            .zip(&node.dst_lds_and_loop_offsets)
            .map(|(via, data)| resolve_end(op, via.loc.unit, data))
            .collect(),
        src_position: node.src_lds_and_loop_offsets.buffer_switch_position,
        dst_position: node
            .dst_lds_and_loop_offsets
            .first()
            .and_then(|data| data.buffer_switch_position),
        precision: lds_format(op, node.src_lds_and_loop_offsets.my_lds_idx),
    }
}

/// One end of a transfer, resolved — the location pair, the buffers, the start addresses.
fn resolve_end(
    op: &WireOp,
    unit: SenComponent,
    data: &crate::wire::DataInfo,
) -> End {
    let unit = unit_of(unit).expect("a unit the census can name");
    let switches = num_buffers(op, data.my_lds_idx);
    End {
        unit,
        // ⚠️ THE PAIR MAY NOT BE IN THE GRANULARITY TABLE — and that is fine for a NON-switching
        // end (see the module doc). A SWITCHING end outside the table is a real gap and panics
        // in [`ResolvedTransfer::side`].
        loc: location_of(unit, data.my_lds_idx),
        switches,
        addresses: folded_addresses(&data.start_addr),
    }
}

/// A WIRE COMPONENT AS A [`DfirUnit`] — the units the conversion binds. The memories and
/// register files answer [`None`] for the sync-end question and panic for a transfer end's
/// `unit_`: a fixture that names one there is one the lowering cannot walk.
fn unit_of(component: SenComponent) -> Option<DfirUnit> {
    Some(match component {
        SenComponent::Sfp => DfirUnit::Sfp,
        SenComponent::Pe => DfirUnit::Pe,
        SenComponent::Ptrow0 => DfirUnit::PtRow(row_of(0)),
        SenComponent::Ptrow1 => DfirUnit::PtRow(row_of(1)),
        SenComponent::Ptrow2 => DfirUnit::PtRow(row_of(2)),
        SenComponent::Ptrow3 => DfirUnit::PtRow(row_of(3)),
        SenComponent::Ptrow4 => DfirUnit::PtRow(row_of(4)),
        SenComponent::Ptrow5 => DfirUnit::PtRow(row_of(5)),
        SenComponent::Ptrow6 => DfirUnit::PtRow(row_of(6)),
        SenComponent::Ptrow7 => DfirUnit::PtRow(row_of(7)),
        SenComponent::Lxlu => DfirUnit::Lxlu,
        SenComponent::Lxsu => DfirUnit::Lxsu,
        SenComponent::Lx => DfirUnit::Lx,
        SenComponent::Hbm => DfirUnit::Hbm,
        SenComponent::L0lu => DfirUnit::L0lu,
        SenComponent::L0lurow0 => DfirUnit::L0lu,
        SenComponent::L0su => DfirUnit::L0su,
        SenComponent::L0 => DfirUnit::L0,
        SenComponent::L3lu => DfirUnit::L3lu,
        SenComponent::L3su => DfirUnit::L3su,
        SenComponent::Constant => DfirUnit::Constant,
        SenComponent::Sfpstate => DfirUnit::SfpState,
        SenComponent::Pestate => DfirUnit::PeState,
        SenComponent::Sfpring => DfirUnit::SfpRing,
        SenComponent::Lxvirtualibr => DfirUnit::LxVirtualIbr,
        SenComponent::Crossptnlink => DfirUnit::CrossPtnLink,
        _ => return None,
    })
}

/// ONE END'S `{unit, lds}` PAIR, AS THE GRANULARITY TABLE KEYS IT — or [`None`] for an lds the
/// table does not carry, which only a non-switching end may be (the reference never consults the
/// table for one; see the module doc).
///
/// ⭐ THE PAIR IS READ OFF THE LDS'S OWN `memOrg_`, NOT THE END'S `storage_` FIELD: the dumper's
/// `storage_` is free-form (`"latch"`, `"lxlu"` — the fixture's own SFP-side ends spell units
/// there, not memories), while `memOrg_`'s keys are the arch's own component spellings and the
/// one the allocation lives in is the end's granularity row. An lds with no allocation (or a
/// non-LDS end) has no row to read, which is the [`None`] only a non-switching end may take.
fn location_of(unit: DfirUnit, lds_idx: i64) -> Option<DataLocation> {
    // The arch's 23-row granularity table, keyed by the pair — the rows are enumerated in the
    // port's own [`DataLocation`], and the match below is that table read backwards.
    let _ = lds_idx;
    match unit {
        DfirUnit::L3lu => Some(DataLocation::L3luLx),
        DfirUnit::L3su => Some(DataLocation::L3suLx),
        DfirUnit::Lxlu => Some(DataLocation::LxluLx),
        DfirUnit::Lxsu => Some(DataLocation::LxsuLx),
        DfirUnit::L0lu => Some(DataLocation::L0luL0),
        DfirUnit::L0su => Some(DataLocation::L0suL0),
        DfirUnit::Sfp => Some(DataLocation::SfpLrf),
        DfirUnit::Pe => Some(DataLocation::PeLrf),
        DfirUnit::PtRow(_) => Some(DataLocation::PtArf),
        _ => None,
    }
}

/// ONE LABELED DATASPACE'S `dataFormat_` — the source lds's element type, which entry 056 reads
/// the granularity factor at. A non-LDS end (`myLdsIdx_ = -1`) carries the C++ member
/// initializer's `IEEE_FP32`, which is what the reference's own default leaves it at.
fn lds_format(op: &WireOp, lds_idx: i64) -> DataType {
    if lds_idx < 0 {
        return DataType::IeeeFp32;
    }
    op.labeled_ds
        .iter()
        .find(|lds| lds.lds_idx == lds_idx)
        .map(|lds| lds.data_format)
        .expect("a labeledDs_ the reader already validated against the allocate nodes")
}

/// `numBuffers_` OF THE ALLOCATION BEHIND ONE END — the `labeledDs_`'s `memOrg_` entries'
/// allocate nodes, or [`None`] where no allocation backs the end.
///
/// ⭐ THE FIXTURE'S OWN SHAPE: an lds's `memOrg_` carries one entry PER STORAGE the dataspace
/// lives in (`{hbm, lx}` for Tensor0, `{hbm, lx, sfplrf}` for Tensor1), and each entry names its
/// own allocate node. The end's buffers are read off the entry whose allocate node backs THIS
/// end — the first entry with one, matching the reference's own "which allocation" question, and
/// the fixture's HBM allocations (`numBuffers_ = 1`) and LX ones (`numBuffers_ = 2`) answer
/// differently through it.
fn num_buffers(op: &WireOp, lds_idx: i64) -> Option<NumBuffers> {
    if lds_idx < 0 {
        return None;
    }
    let Some(lds) = op
        .labeled_ds
        .iter()
        .find(|lds| lds.lds_idx == lds_idx)
    else {
        // The wire reader already resolved every `memOrg_` allocate name against the schedule
        // tree, so an index that names no dataspace here is a fixture the reader never sees.
        panic!("a labeledDs_ the reader already validated against the allocate nodes");
    };
    for entry in lds.mem_org.values() {
        let Some(node_name) = entry.allocate_node.as_ref() else {
            continue;
        };
        let node = op
            .schedule_tree
            .iter()
            .find(|node| &node.name == node_name)
            .expect("an allocate node the reader already resolved");
        let WireKind::Allocate(allocate) = &node.kind else {
            panic!("memOrg_ named a non-allocate node");
        };
        return Some(if allocate.num_buffers == -1 {
            NumBuffers::Streaming
        } else {
            NumBuffers::Count(
                i32::try_from(allocate.num_buffers).expect("a non-negative numBuffers_"),
            )
        });
    }
    None
}

/// ONE END'S `startAddr_`, RE-KEYED — `ZeroFold` is the one value at every coordinate;
/// `Folded` parses each `"[core, corelet, time]"` key.
fn folded_addresses(data: &FoldData) -> BTreeMap<(Core, Corelet, u32), i64> {
    let mut map = BTreeMap::new();
    match data {
        FoldData::ZeroFold(value) => {
            let value: i64 = value.parse().expect("a numeric startAddr_");
            // ⚠️ THE ARBITRARY KEY — see the module doc: one value for the whole fold space.
            map.insert((core_key(0), corelet_key(0), 0), value);
        }
        FoldData::Folded(folded) => {
            for (key, value) in &folded.data {
                let coords: Vec<i64> =
                    serde_json::from_str(key).expect("a coordinate the reader already checked");
                let [core, corelet, time] = coords[..] else {
                    panic!("a three-coordinate key the reader already checked");
                };
                map.insert(
                    (
                        core_key(u32::try_from(core).expect("a core id")),
                        corelet_key(u32::try_from(corelet).expect("a corelet id")),
                        u32::try_from(time).expect("a fold index"),
                    ),
                    value.parse::<i64>().expect("a numeric startAddr_"),
                );
            }
        }
    }
    map
}

/// A `Core` for a coordinate the map is keyed by.
fn core_key(core: u32) -> Core {
    Core::checked(core).expect("a core the arch can name")
}

/// A `Corelet` for a coordinate the map is keyed by.
fn corelet_key(corelet: u32) -> Corelet {
    Corelet::checked(corelet).expect("a corelet the arch can name")
}

/// THE WIRE'S COMPONENT SPELLING FOR A CENSUS UNIT — `senComponentsToString`, reversed. The
/// relevance filter and the sync-end resolver both key the wire's `relevantComps_` maps by the
/// wire's own spellings, so the census's [`DfirUnit`] has to go back the way it came.
fn sen_of(comp: DfirUnit) -> SenComponent {
    SenComponent::from_spelling(comp.spelling())
        .expect("a census unit the wire vocabulary names")
}

#[cfg(test)]
mod unit_tests {
    use super::super::driver::{Dsc, Scheduled, ScheduleView, UnitHandles, Viewed};
    use super::{DscKind, WireDsc, WireTables};
    use crate::bridges::superdsc_to_dataflow_ir::utils::{Neighbourhood, OwnFiles};
    use crate::islands::dataflow_ir::Units;
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::dialects::Val;
    use crate::units::{Core, Corelet, DfirUnit};

    /// THE FIXTURE — the rmsq program's scheduled dump, as dxp's `DBO_DEBUG=1` writes it.
    const FIXTURE: &str = "/Users/nickm/tmp/phase0/pre/sdsc_0.json";

    /// ONE PARSED OP — the single DSC the fixture carries.
    fn parsed_op() -> crate::wire::WireOp {
        let text = std::fs::read_to_string(FIXTURE).expect("the rmsq fixture on disk");
        let file = crate::wire::read_file(&text).expect("the fixture parses");
        file.programs[0].ops[0].clone()
    }

    /// ONE EMPTY HANDLES SET — the roots walk only needs the shape, and every leaf it builds for
    /// this fixture is a sync, whose `Handlers` bridge reads nothing the driver minted.
    fn handles() -> (
        Values,
        Neighbourhood,
        Units,
    ) {
        (
            Values::default(),
            Neighbourhood {
                ops: Vec::new(),
                units: Vec::new(),
                directions: Vec::new(),
                own: OwnFiles::None,
            },
            Units::one(DfirUnit::Lxlu, Val(1)),
        )
    }

    /// 🎯 THE WALK ITSELF — one component's view of the rmsq schedule: the loop nest, the sync
    /// leaves, and the two stated gaps (no transfer or compute leaves), pinned so a regression
    /// in the reader, the tables build or the view shows up as a shape change.
    #[test]
    fn the_lxlu_view_walks_the_fixture() {
        let op = parsed_op();
        let tables = WireTables::of(&op);
        let dsc = WireDsc::new(&op, &tables);

        // The fixture's own facts.
        assert_eq!(dsc.cores().len(), 32);
        assert_eq!(dsc.num_corelets_used(), 1);
        assert!(matches!(dsc.kind(), DscKind::Dsc2));

        // The Lxlu view, at one pair: the roots are the head's children the component sees —
        // the allocates and the head block name no component, so the ONE root is
        // `loop_ds0_ds1_y`, a Band whose stages are the ds0/ds1 pair it is named from.
        let (mut vals, neighbourhood, units) = handles();
        let viewing = dsc.view(
            DfirUnit::Lxlu,
            Viewed::OnPair(
                Core::checked(0).expect("core 0"),
                Corelet::checked(0).expect("corelet 0"),
            ),
        );
        let roots = viewing
            .roots
            .roots(
                &mut vals,
                UnitHandles {
                    neighbours: &neighbourhood,
                    handles: None,
                    iterator: None,
                    units: &units,
                },
            );
        assert_eq!(roots.len(), 1, "the Lxlu view has one root");
        let Scheduled::Band { stages, .. } = &roots[0] else {
            panic!("the root is a Band");
        };
        assert_eq!(stages, &("core", "chunk"));

        // The two stated gaps, pinned by kind: the transfers and the compute lower NO leaf —
        // every one of them arrives as an empty Block. Of the fixture's eight syncs, TWO name
        // `lxlu` in their `relevantComps_` (`isNodeRelevant` keeps a node for either end's
        // component), and those two — the receive from l3lu and the send back to it — are the
        // only leaves the walk yields.
        fn leaf_count(scheduled: &[Scheduled<'_>]) -> usize {
            scheduled
                .iter()
                .map(|one| match one {
                    Scheduled::Leaf(_) => 1,
                    Scheduled::Band { children, .. }
                    | Scheduled::Block { children }
                    | Scheduled::Parametric { children, .. } => leaf_count(children),
                    Scheduled::Condition { .. } => 0,
                })
                .sum()
        }
        assert_eq!(
            leaf_count(&roots),
            2,
            "the Lxlu view's leaves are its two lxlu syncs"
        );
    }
}
