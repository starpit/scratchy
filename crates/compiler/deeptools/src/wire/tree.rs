//! THE SCHEDULE TREE — `dsc/dsc2.cpp:1300-1870`'s import, as typed data.
//!
//! The C++ import is TWO SCANS, and the reader mirrors both:
//!
//! 1. **Scan 1** (`:1327-1397`) creates every node, refuses a missing `nodeType_`/`name_`/`prev_`
//!    ("Missing basic fields"), refuses a duplicate name, links `prev_` through the name map (an
//!    empty `prev_` is the head's child; a missing one is "Illegal scheduleTree_, cannot import"),
//!    and fills the head's `relevantComps_` as the union of its children's.
//! 2. **Scan 2** (`:1399-1870`) walks the array AGAIN and imports each node's own fields, per kind.
//!
//! `next_` IS NOT IMPORTED — the tree is rebuilt from `prev_` alone (`:1389` reads
//! `getHeadMutable()->next_`, which the linking filled), so the reader ignores the dumper's
//! `next_` arrays entirely and re-derives children the same way.
//!
//! After both scans the importer runs two cross-checks the reader also runs:
//!
//! - every `allocUsers_` name must be a node ("Usernode X of AllocateNode Y was not found. Illegal
//!   scheduleTree_, cannot import", `:1846-1849`);
//! - an allocate node's `ldsIdx_ >= 0` names a `labeledDs_` entry, and that entry's
//!   `memOrg_[component_].allocateNode_` is set to the node (`:1852-1859`).

use std::collections::BTreeMap;

use serde::Deserialize;
use sys_arch_spec::arch_enums::SenComponent;

use super::coords::Coordinates;
use super::enums::{IndirectAllocType, MetaDimKind, WireComputeType};
use super::fold::FoldData;
use super::{PrimaryDim, Refusal};

/// ONE NODE OF THE SCHEDULE TREE — the scan-1 shape every kind shares, plus its own fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleNode {
    /// `name_` — the tree's identity key. Duplicate names are refused in scan 1.
    pub name: String,
    /// `prev_` — the parent's name. Empty means the head's child (`dsc/dsc2.cpp:1343-1348`).
    pub prev: String,
    /// `relevantComps_` — component → core → corelet ids (`:1369-1385`).
    pub relevant_comps: BTreeMap<SenComponent, BTreeMap<i64, Vec<i64>>>,
    /// The node's own fields, per kind.
    pub kind: NodeKind,
}

/// ONE NODE'S OWN FIELDS — the scan-2 import, per `nodeType_` (`dsc/dsc2.cpp:1403-1870`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeKind {
    /// `block` — nothing beyond the scan-1 shape (`:1428`'s implicit fall-through).
    Block,
    /// `loop` (`:1403-1427`).
    Loop(LoopNode),
    /// `transfer` (`:1444-1636`).
    Transfer(TransferNode),
    /// `compute` (`:1638-1714`).
    Compute(ComputeNode),
    /// `sync` (`:1716-1750`).
    Sync(SyncNode),
    /// `allocate` (`:1752-1850`).
    Allocate(AllocateNode),
}

/// A LOOP NODE — scan 2's LOOP arm (`dsc/dsc2.cpp:1403-1427`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopNode {
    /// `numId_`.
    pub num_id: i64,
    /// `denId_`.
    pub den_id: i64,
    /// `parametricLoop_`.
    pub parametric: bool,
    /// `parametricLdsIdx_`.
    pub parametric_lds_idx: i64,
    /// `dims_` — one `{dim_, kind_}` per dimension the loop runs over.
    pub dims: Vec<LoopDim>,
    /// `loopCountSymbolIds_`.
    pub loop_count_symbol_ids: BTreeMap<i64, i64>,
}

/// ONE `dims_` ENTRY — `newDim.dim_`/`newDim.kind_` (`dsc/dsc2.cpp:1416-1423`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopDim {
    /// `dim_`.
    pub dim: PrimaryDim,
    /// `kind_`.
    pub kind: MetaDimKind,
}

/// A TRANSFER NODE — scan 2's TRANSFER arm (`dsc/dsc2.cpp:1444-1636`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferNode {
    /// `src_` — `{unit_, storage_}`.
    pub src: Loc,
    /// `srcIndirect_` — present only where the source is an indirect access.
    pub src_indirect: Option<Loc>,
    /// `dstVias_` — the destinations, each with its route.
    pub dst_vias: Vec<DstVia>,
    /// `lastFusableParentLoopSrc_` — a node name, resolved by the reader to the node's index.
    pub last_fusable_parent_loop_src: Option<usize>,
    /// `lastFusableParentLoopDst_` — node names, one per destination.
    pub last_fusable_parent_loop_dst: Vec<Option<usize>>,
    /// `srcLdsAndLoopOffsets_`.
    pub src_lds_and_loop_offsets: DataInfo,
    /// `srcIndirectLdsAndLoopOffsets_`.
    pub src_indirect_lds_and_loop_offsets: Option<DataInfo>,
    /// `dstLdsAndLoopOffsets_` — one per destination.
    pub dst_lds_and_loop_offsets: Vec<DataInfo>,
    /// `dstIndirectLdsAndLoopOffsets_` — present only where a destination is indirect.
    pub dst_indirect_lds_and_loop_offsets: Option<Vec<DataInfo>>,
    /// `replicationFactor_`.
    pub replication_factor: i64,
    /// `unitTimeTransferChunkSize_` — `{sizeDim_, srcSizeIdx_, dstSizeIdx_}`.
    pub chunk_size: Vec<ChunkSize>,
    /// `unitTimeTransferChunkStride_` — same shape.
    pub chunk_stride: Vec<ChunkSize>,
    /// `unitTimeTransferNumChunks_`.
    pub num_chunks: i64,
    /// `rotateNumElements_`.
    pub rotate_num_elements: i64,
    /// `coreIdToGTRInfo_` — core → `{groupId_, numSharers_}`.
    pub core_id_to_gtr_info: BTreeMap<i64, GtrInfo>,
    /// `transferSize_` — dim → int.
    pub transfer_size: BTreeMap<PrimaryDim, i64>,
    /// `coreletViews_` — corelet → the per-corelet view of the transfer.
    pub corelet_views: BTreeMap<i64, CoreletView>,
    /// `coordinates_`.
    pub coordinates: Option<Coordinates>,
}

/// A `{unit_, storage_}` PAIR — the transfer's endpoints (`dsc/dsc2.cpp:1466-1480`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Loc {
    /// `unit_`.
    pub unit: SenComponent,
    /// `storage_`.
    pub storage: SenComponent,
}

/// ONE `dstVias_` ENTRY (`dsc/dsc2.cpp:1483-1500`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DstVia {
    /// `loc_` — the destination.
    pub loc: Loc,
    /// `locIndirect_` — present only where the destination is indirect.
    pub loc_indirect: Option<Loc>,
    /// `via_` — the units the data routes through.
    pub via: Vec<SenComponent>,
}

/// ONE `unitTimeTransferChunkSize_`/`ChunkStride_` ENTRY (`dsc/dsc2.cpp:1535-1560`). The dumper
/// nests the whole `Size` under `sizeDim_` (`dsc/dsc2.cpp:593-595`), so the size value travels
/// with the dim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkSize {
    /// `sizeDim_` — the whole Size object.
    pub size_dim: Size,
    /// `srcSizeIdx_`.
    pub src_size_idx: i64,
    /// `dstSizeIdx_`.
    pub dst_size_idx: i64,
}

/// ONE `coreIdToGTRInfo_` ENTRY (`dsc/dsc2.cpp:1526-1534`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GtrInfo {
    /// `groupId_`.
    pub group_id: i64,
    /// `numSharers_`.
    pub num_sharers: i64,
}

/// ONE CORELET'S VIEW OF A TRANSFER — `coreletViews_` (`dsc/dsc2.cpp:1563-1600`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreletView {
    /// `srcLoopsAndSize_`.
    pub src_loops_and_size: UnitView,
    /// `srcIndirectLoopsAndSize_`.
    pub src_indirect_loops_and_size: UnitView,
    /// `dstLoopsAndSizes_` — one per destination.
    pub dst_loops_and_sizes: Vec<UnitView>,
    /// `dstIndirectLoopsAndSizes_`.
    pub dst_indirect_loops_and_sizes: Vec<UnitView>,
}

/// A `DataInfo` — one operand's placement (`dsc/dsc2.cpp:1218-1250`, `importDataInfo`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataInfo {
    /// `myLdsIdx_` — the labeled dataspace, or -1 for a non-LDS operand (e.g. `zero`).
    pub my_lds_idx: i64,
    /// `startAddr_` — a fold value.
    pub start_addr: FoldData,
    /// `isStartAddrSymbolic_`.
    pub is_start_addr_symbolic: bool,
    /// `latchDataId_`.
    pub latch_data_id: i64,
    /// `constantId_`.
    pub constant_id: i64,
    /// `constEleOffsets_` — core → corelet → dim → offset.
    pub const_ele_offsets: BTreeMap<i64, BTreeMap<i64, BTreeMap<PrimaryDim, i64>>>,
    /// `loopEleOffsets_` — corelet → loop node name → dim → offset.
    pub loop_ele_offsets: BTreeMap<i64, BTreeMap<String, BTreeMap<PrimaryDim, i64>>>,
    /// `bufferAddrOffset_` — core → corelet → offset (i64-as-string on the wire).
    pub buffer_addr_offset: BTreeMap<i64, BTreeMap<i64, i64>>,
    /// `bufferSwitchPosition_` — a loop node name, or empty.
    pub buffer_switch_position: Option<usize>,
    /// `dataConnect_`.
    pub data_connect: Option<DataConnect>,
}

/// A `dataConnect_` — WHICH PORT AN OPERAND ARRIVES ON. The C++ reads it as a free-form string
/// (`importDataInfo`, `dsc/dsc2.cpp:1247`); the `.ddl` census closed it into
/// [`crate::generated::DataConnect`] but the WIRE's vocabulary is the census's lower-case
/// spellings plus the dumper's own empty-string-for-none, so the reader keeps the string and the
/// adapter closes it per consumer.
///
/// ⚠️ THE EMPTY STRING IS NOT A MISSING FIELD. The dumper always writes `dataConnect_` (the
/// fixtures carry `""` for the `zero` operand), so absence parses to [`None`] here only when the
/// key itself is absent, and the empty string is [`Some`] carrying nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataConnect(pub String);

impl DataConnect {
    /// The wire's spelling, or the empty string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// ONE `importUnitView` — `sizesNoGaps_`, `compositeLoops_`, `outerLoops_`, `sizesWithGaps_`
/// (`dsc/dsc2.cpp:1252-1265`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UnitView {
    /// `sizesNoGaps_` — one `Size` per access dimension.
    pub sizes_no_gaps: Vec<Size>,
    /// `compositeLoops_`.
    pub composite_loops: Vec<LoopInfo>,
    /// `outerLoops_`.
    pub outer_loops: Vec<LoopInfo>,
    /// `sizesWithGaps_` — core → sizes.
    pub sizes_with_gaps: BTreeMap<i64, Vec<Size>>,
}

/// A `Size` — `importSize`'s `{dim_, size_}` (`dsc/dsc2.cpp:1214-1217`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Size {
    /// `dim_`.
    pub dim: PrimaryDim,
    /// `size_`.
    pub size: i64,
}

/// ONE `LoopInfo` — `importLoopInfo`'s `{loop_, dim_, sizeIdx_, elemOffset_}`
/// (`dsc/dsc2.cpp:1219-1224`). `loop_` is a node name resolved to the node's index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopInfo {
    /// `loop_` — the loop node's index in the tree.
    pub loop_idx: usize,
    /// `dim_`.
    pub dim: PrimaryDim,
    /// `sizeIdx_`.
    pub size_idx: i64,
    /// `elemOffset_`.
    pub elem_offset: i64,
}

/// A COMPUTE NODE — scan 2's COMPUTE arm (`dsc/dsc2.cpp:1638-1714`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeNode {
    /// `exUnit_`.
    pub ex_unit: SenComponent,
    /// `type_`.
    pub ty: WireComputeType,
    /// `dataFormat_`.
    pub data_format: crate::generated::DataType,
    /// `inputs_`.
    pub inputs: Vec<SenComponent>,
    /// `outputs_`.
    pub outputs: Vec<SenComponent>,
    /// `inputsLdsAndLoopOffsets_` — one per input.
    pub inputs_lds_and_loop_offsets: Vec<DataInfo>,
    /// `outputsLdsAndLoopOffsets_` — one per output.
    pub outputs_lds_and_loop_offsets: Vec<DataInfo>,
    /// `coreletViews_` — corelet → the per-corelet view.
    pub corelet_views: BTreeMap<i64, ComputeCoreletView>,
    /// `numFoldsEngaged`.
    pub num_folds_engaged: i64,
    /// `isOpaqueOp_`.
    pub is_opaque_op: bool,
    /// `instrAttribute_` — free-form in the C++ (`:1683-1685`); kept as the string it arrives as.
    pub instr_attribute: BTreeMap<String, String>,
    /// `inputCoordinates_` — position → coordinates.
    pub input_coordinates: BTreeMap<i64, Coordinates>,
    /// `outputCoordinates_`.
    pub output_coordinates: Option<Coordinates>,
}

/// ONE CORELET'S VIEW OF A COMPUTE (`dsc/dsc2.cpp:1687-1702`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeCoreletView {
    /// `inputsLoopsAndSizes_` — one per input.
    pub inputs_loops_and_sizes: Vec<UnitView>,
    /// `outputsLoopsAndSizes_` — one per output.
    pub outputs_loops_and_sizes: Vec<UnitView>,
}

/// A SYNC NODE — scan 2's SYNC arm (`dsc/dsc2.cpp:1716-1750`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncNode {
    /// `units_`.
    pub units: Vec<SenComponent>,
    /// `isReceive_`.
    pub is_receive: bool,
    /// `isSoft_`.
    pub is_soft: bool,
    /// `implicitSyncRefTransfer_` — a transfer node name, resolved to its index.
    pub implicit_sync_ref_transfer: Option<usize>,
    /// `otherEndOfTheSignals_` — sync node names, resolved to their indices.
    pub other_end_of_the_signals: Vec<usize>,
}

/// AN ALLOCATE NODE — scan 2's ALLOCATE arm (`dsc/dsc2.cpp:1752-1850`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocateNode {
    /// `ldsIdx_`.
    pub lds_idx: i64,
    /// `constIdx_`.
    pub const_idx: i64,
    /// `tempStorageForCompute_` — a compute node name, resolved to its index.
    pub temp_storage_for_compute: Option<usize>,
    /// `component_`.
    pub component: SenComponent,
    /// `padding_` — dim → pad type.
    pub padding: BTreeMap<PrimaryDim, super::enums::PadType>,
    /// `layoutDimOrder_`.
    pub layout_dim_order: Vec<PrimaryDim>,
    /// `maxDimSizes_` — one per `layoutDimOrder_` entry.
    pub max_dim_sizes: Vec<i64>,
    /// `numBuffers_`.
    pub num_buffers: i64,
    /// `startAddressCoreCorelet_` — a fold value.
    pub start_address_core_corelet: FoldData,
    /// `isStartAddrSymbolic_`.
    pub is_start_addr_symbolic: bool,
    /// `bufferOffsetCoreCorelet_` — core → corelet → offset (i64-as-string on the wire).
    pub buffer_offset_core_corelet: BTreeMap<i64, BTreeMap<i64, i64>>,
    /// `backGapCore_` — dim → core → gap (i64-as-string on the wire).
    pub back_gap_core: BTreeMap<PrimaryDim, BTreeMap<i64, i64>>,
    /// `indirectAllocType_`.
    pub indirect_alloc_type: IndirectAllocType,
    /// `indexTensorType_` — only where `indirectAllocType_` is `index_tensor`.
    pub index_tensor_type: Option<String>,
    /// `relatedIndirectAccessAlloc_` — an allocate node name, resolved to its index.
    pub related_indirect_access_alloc: Option<usize>,
    /// `ignoreSymbolicVolumeLimits_`.
    pub ignore_symbolic_volume_limits: bool,
    /// `nonUnifiedAllocInHBM_`.
    pub non_unified_alloc_in_hbm: bool,
    /// `gapStickSpread_` — dim → spread.
    pub gap_stick_spread: BTreeMap<PrimaryDim, i64>,
    /// `allocUsers_` — node name → reference count. DANGLING NAMES ARE REFUSED
    /// (`dsc/dsc2.cpp:1846-1849`), which the reader enforces after the scan.
    pub alloc_users: BTreeMap<String, i64>,
    /// `coordinates_`.
    pub coordinates: Option<Coordinates>,
}

// -- serde shapes --------------------------------------------------------------

/// A `{unit_, storage_}` pair's wire form.
#[derive(Deserialize)]
pub(super) struct WireLoc {
    #[serde(rename = "unit_")]
    unit: String,
    #[serde(rename = "storage_")]
    storage: String,
}

impl TryFrom<WireLoc> for Loc {
    type Error = Refusal;

    fn try_from(wire: WireLoc) -> Result<Self, Refusal> {
        Ok(Self {
            unit: SenComponent::from_spelling(&wire.unit).ok_or(Refusal::UnknownSpelling {
                spelling: wire.unit,
                field: "unit_",
            })?,
            storage: SenComponent::from_spelling(&wire.storage).ok_or(
                Refusal::UnknownSpelling {
                    spelling: wire.storage,
                    field: "storage_",
                },
            )?,
        })
    }
}

/// A component → core → corelet-ids map's wire form.
pub(super) type WireRelevantComps = BTreeMap<String, BTreeMap<String, Vec<i64>>>;

pub(super) fn relevant_comps(wire: WireRelevantComps) -> Result<BTreeMap<SenComponent, BTreeMap<i64, Vec<i64>>>, Refusal> {
    let mut out = BTreeMap::new();
    for (comp_str, cores) in wire {
        let comp = SenComponent::from_spelling(&comp_str).ok_or(Refusal::UnknownSpelling {
            spelling: comp_str,
            field: "relevantComps_",
        })?;
        let mut per_core = BTreeMap::new();
        for (core_str, corelets) in cores {
            let core: i64 = core_str.parse().map_err(|_| Refusal::MalformedInteger {
                text: core_str,
                field: "relevantComps_ core".into(),
            })?;
            per_core.insert(core, corelets);
        }
        out.insert(comp, per_core);
    }
    Ok(out)
}
