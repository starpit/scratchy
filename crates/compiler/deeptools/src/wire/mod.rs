//! THE SCHEDULED-SUPERDSC WIRE READER — dbo's `dumpProgram` JSON, parsed.
//!
//! # WHAT THIS READS
//!
//! The file `DBO_DEBUG=1 dxp_standalone --export-dir=<dir> --bundle ... -b sentient` leaves at
//! `<export_dir>/debug/<name>/sdsc.json`: a SuperDSC *after* the C++ scheduler has run, which is
//! the input the SuperDSC→DataflowIR lowering consumes (`dsc/dsc2.cpp:1300-1870`'s import is the
//! C++'s own reader of the same shape). This module is that reader, in Rust, for the
//! C++-schedules/Rust-lowers split.
//!
//! # THE CONTRACT
//!
//! - **Closed sets are enums at the boundary.** Every string the C++ parses through an
//!   `EnumsConversion` table parses here through a `from_spelling`, re-using
//!   [`sys_arch_spec::arch_enums::SenComponent`] and [`crate::generated::DataType`] where the
//!   vocabulary already exists. Nothing downstream sees a raw string standing in for a kind.
//! - **A malformed program is a refusal, not a wrong lowering.** [`read_program`] returns
//!   `Result<_, Refusal>`, and [`Refusal`] carries the field that failed. The C++ raises
//!   `DT_ERROR` for the same shapes; a refusal here is that error with its evidence kept.
//! - **The two scans are two passes here too.** Names are collected, linked and validated before
//!   any per-kind field is read, exactly as `dsc/dsc2.cpp:1327-1397` does before `:1399-1870`.
//!
//! # WHAT THIS DELIBERATELY DOES NOT DO
//!
//! - It does not lower. Producing [`crate::islands::dataflow_ir`] values from this tree is the
//!   adapter's job (bridge 1's `ScheduleView`), and it lives on the bridge side of the boundary.
//! - It does not interpret fold values. [`fold::FoldData`] keeps the wire's strings; which are
//!   integers and which are symbol names is a per-field fact (`isStartAddrSymbolic_`) the adapter
//!   reads.
//! - It ignores `next_`. The tree is rebuilt from `prev_` alone, the way the C++ importer does.

mod coords;
mod enums;
mod fold;
mod tree;

pub use coords::{CoordInfo, Coordinates};
pub use enums::{
    data_format, IndirectAllocType, LdsSegment, MetaDimKind, NodeType, PadType, SenTarget,
    WireComputeType,
};
pub use fold::{FoldData, FoldDimFunc, FoldDimProp, FoldedData};
pub use tree::{
    AllocateNode, ChunkSize, ComputeCoreletView, ComputeNode, CoreletView, DataConnect, DataInfo,
    DstVia, GtrInfo, Loc, LoopDim, LoopInfo, LoopNode, NodeKind, ScheduleNode, Size, SyncNode,
    TransferNode, UnitView,
};

use std::collections::BTreeMap;

use serde::Deserialize;
use sys_arch_spec::arch_enums::SenComponent;

pub use crate::bridges::superdsc_to_dataflow_ir::control_flow::PrimaryDim;

/// WHY A WIRE FILE WAS REFUSED — one variant per `DT_ERROR` the C++ importer can raise, with the
/// evidence that triggered it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// A node is missing `nodeType_`, `name_` or `prev_` — `"Missing basic fields"`
    /// (`dsc/dsc2.cpp:1330-1332`).
    MissingBasicFields {
        /// The node's index in `scheduleTree_`.
        index: usize,
    },
    /// A `nodeType_` the importer's `stringToNodeType` cannot map (`:1333-1336`).
    UnknownNodeType {
        /// The spelling.
        spelling: String,
    },
    /// A duplicate `name_` — `"Schedule Tree has nodes with duplicate name"` (`:1360-1363`).
    DuplicateName {
        /// The duplicated name.
        name: String,
    },
    /// A `prev_` that names no node — `"Illegal scheduleTree_, cannot import"` (`:1343-1348`).
    UnknownParent {
        /// The `prev_` spelling.
        name: String,
        /// The child that carried it.
        child: String,
    },
    /// An `allocUsers_` name that is no node — `"Usernode X of AllocateNode Y was not found.
    /// Illegal scheduleTree_, cannot import"` (`dsc/dsc2.cpp:1846-1849`).
    DanglingAllocUser {
        /// The missing user.
        user: String,
        /// The allocate node that named it.
        alloc: String,
    },
    /// An unknown spelling in a closed set, with the field it was read for.
    UnknownSpelling {
        /// The spelling.
        spelling: String,
        /// The field.
        field: &'static str,
    },
    /// An unknown dimension spelling, with the field it was read for.
    UnknownDim {
        /// The spelling.
        spelling: String,
        /// The field.
        field: &'static str,
    },
    /// An integer that arrived as a string that does not parse — the `ImportUtil::importData`
    /// integral arm's `DT_ERROR` (`util/import_utils.h:39-46`).
    MalformedInteger {
        /// The string.
        text: String,
        /// The field.
        field: String,
    },
    /// A fold object whose two dimension arrays disagree — `"Different number of dims between
    /// json and caller"` (`util/foldManager/foldInfrastructure.h:2779-2780`).
    FoldDimMismatch {
        /// `dim_prop_func`'s length.
        funcs: usize,
        /// `dim_prop_attr`'s length.
        props: usize,
    },
    /// A `data_` key that is not a JSON int array.
    MalformedCoordinate {
        /// The key.
        key: String,
    },
    /// A `data_` coordinate whose length is not the fold count — `"Num of dimensions in coordinate
    /// not matching num folds"` (`util/foldManager/foldInfrastructure.h:2817-2818`).
    FoldCoordMismatch {
        /// The key.
        key: String,
        /// The coordinate's length.
        coord: usize,
        /// The fold count.
        folds: usize,
    },
    /// A fold dimension function the importer does not accept (`:2793-2795`).
    UnknownFoldFunc,
    /// A negative `factor_`, which cannot be a fold cardinality.
    NegativeFoldFactor,
    /// A fold object of a shape the three-key reader cannot parse.
    MalformedFold {
        /// serde's message.
        message: String,
    },
    /// A `coordinates_` of a shape the reader cannot parse.
    MalformedCoordinates {
        /// serde's message.
        message: String,
    },
    /// The file is not the name-keyed object the dumper writes.
    MalformedTopLevel {
        /// serde's message.
        message: String,
    },
    /// A program entry of a shape the reader cannot parse.
    MalformedProgram {
        /// serde's message.
        message: String,
    },
    /// An op of a shape the reader cannot parse.
    MalformedOp {
        /// serde's message.
        message: String,
    },
    /// A schedule-tree node of a shape the reader cannot parse.
    MalformedNode {
        /// serde's message.
        message: String,
    },
}

impl core::fmt::Display for Refusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingBasicFields { index } => {
                write!(f, "node {index}: missing nodeType_/name_/prev_")
            }
            Self::UnknownNodeType { spelling } => write!(f, "unknown nodeType_ {spelling:?}"),
            Self::DuplicateName { name } => write!(f, "duplicate schedule-tree name {name:?}"),
            Self::UnknownParent { name, child } => {
                write!(f, "node {child:?} names prev_ {name:?}, which is no node")
            }
            Self::DanglingAllocUser { user, alloc } => {
                write!(f, "usernode {user:?} of allocate node {alloc:?} was not found")
            }
            Self::UnknownSpelling { spelling, field } => {
                write!(f, "unknown spelling {spelling:?} for {field}")
            }
            Self::UnknownDim { spelling, field } => {
                write!(f, "unknown dimension {spelling:?} for {field}")
            }
            Self::MalformedInteger { text, field } => {
                write!(f, "malformed integer {text:?} for {field}")
            }
            Self::FoldDimMismatch { funcs, props } => {
                write!(f, "fold has {funcs} funcs but {props} dims")
            }
            Self::MalformedCoordinate { key } => write!(f, "malformed fold coordinate {key:?}"),
            Self::FoldCoordMismatch { key, coord, folds } => {
                write!(f, "fold coordinate {key:?} has {coord} dims, want {folds}")
            }
            Self::UnknownFoldFunc => write!(f, "fold func type not supported in import"),
            Self::NegativeFoldFactor => write!(f, "negative fold factor_"),
            Self::MalformedFold { message } => write!(f, "malformed fold data: {message}"),
            Self::MalformedCoordinates { message } => {
                write!(f, "malformed coordinates_: {message}")
            }
            Self::MalformedTopLevel { message } => write!(f, "malformed sdsc.json: {message}"),
            Self::MalformedProgram { message } => write!(f, "malformed program: {message}"),
            Self::MalformedOp { message } => write!(f, "malformed op: {message}"),
            Self::MalformedNode { message } => write!(f, "malformed schedule node: {message}"),
        }
    }
}

impl std::error::Error for Refusal {}

/// THE WHOLE FILE — one entry per program, keyed by name (`{ "0_rmsq_o728": {...} }`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireFile {
    /// The programs, in file order.
    pub programs: Vec<WireProgram>,
}

/// ONE PROGRAM — the sdsc-level fields the lowering reads, plus the ops.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireProgram {
    /// The key the file named the program by.
    pub name: String,
    /// `numCoresUsed_`.
    pub num_cores_used: i64,
    /// `coreIdToDsc_` — core → index into `dscs_` (`dsc/superdsc.cpp:826-836`; a negative index
    /// is the C++'s "no dsc", so the index is optional).
    pub core_id_to_dsc: BTreeMap<i64, Option<usize>>,
    /// `target_`.
    pub target: SenTarget,
    /// `dscs_` — the ops, which for a scheduled SuperDSC is one entry carrying its own key.
    pub ops: Vec<WireOp>,
}

/// ONE OP — the `dscs_` entry's inner object (`rmsq_o728` in the fixture).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireOp {
    /// The op's own name (the entry's key).
    pub name: String,
    /// `numCoresUsed_`.
    pub num_cores_used: i64,
    /// `numCoreletsUsed_`.
    pub num_corelets_used: i64,
    /// `coreIdsUsed_`.
    pub core_ids_used: Vec<i64>,
    /// `numCoreletsUsed_DSC2_` — read by entry 109 of the port.
    pub num_corelets_used_dsc2: i64,
    /// `scheduleTree_` — the nodes, in file order.
    pub schedule_tree: Vec<ScheduleNode>,
    /// `labeledDs_` — the labeled dataspaces the allocate nodes and DataInfos index into.
    pub labeled_ds: Vec<LabeledDs>,
}

/// ONE `labeledDs_` ENTRY — the fields of `importLabeledDs` the tree's own validation reads
/// (`dsc/dsc2.cpp:1852-1859`: the `memOrg_` → `allocateNode_` back-pointer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabeledDs {
    /// `ldsIdx_`.
    pub lds_idx: i64,
    /// `dsName_`.
    pub ds_name: String,
    /// `dsType_` — the segment kind as the dumper writes it (`OUTPUT`/`INPUT`/...).
    pub ds_type: String,
    /// `segment_` — optional: the dumper writes it only in the full labeledDs form
    /// (`dsc/dataOpDsc.cpp:1347-1349`); some dumps omit it and the C++ import leaves the default.
    pub segment: Option<LdsSegment>,
    /// `memOrg_` — component → the allocation entry, whose `allocateNode_` is a node NAME the
    /// reader resolves during the allocate scan.
    pub mem_org: BTreeMap<SenComponent, MemOrgEntry>,
}

/// ONE `memOrg_` ENTRY.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemOrgEntry {
    /// `isPresent`.
    pub is_present: bool,
    /// `isPadded`.
    pub is_padded: bool,
    /// `isZeroPadded`.
    pub is_zero_padded: bool,
    /// `zpadGapFront`.
    pub zpad_gap_front: Vec<i64>,
    /// `gapPerDim` — dim → gap.
    pub gap_per_dim: BTreeMap<PrimaryDim, i64>,
    /// `dsOffset`.
    pub ds_offset: i64,
    /// `allocateNode_` — the node name; resolved and validated against the tree.
    pub allocate_node: Option<String>,
}

// -- serde shapes --------------------------------------------------------------

/// The file's top-level shape: `{name: program}`.
type WireTopLevel = BTreeMap<String, WireProgramJson>;

/// The program's serde shape.
#[derive(Deserialize)]
struct WireProgramJson {
    #[serde(rename = "numCoresUsed_")]
    num_cores_used: i64,
    #[serde(rename = "coreIdToDsc_", default)]
    core_id_to_dsc: BTreeMap<String, i64>,
    #[serde(rename = "target_", default = "default_target")]
    target: String,
    #[serde(rename = "dscs_", default)]
    dscs: Vec<BTreeMap<String, WireOpJson>>,
}

/// The op's serde shape.
#[derive(Deserialize)]
struct WireOpJson {
    #[serde(rename = "numCoresUsed_", default)]
    num_cores_used: i64,
    #[serde(rename = "numCoreletsUsed_", default)]
    num_corelets_used: i64,
    #[serde(rename = "coreIdsUsed_", default)]
    core_ids_used: Vec<i64>,
    #[serde(rename = "numCoreletsUsed_DSC2_", default)]
    num_corelets_used_dsc2: i64,
    #[serde(rename = "scheduleTree_", default)]
    schedule_tree: Vec<serde_json::Value>,
    #[serde(rename = "labeledDs_", default)]
    labeled_ds: Vec<WireLabeledDsJson>,
}

/// `labeledDs_`'s serde shape.
#[derive(Deserialize)]
struct WireLabeledDsJson {
    #[serde(rename = "ldsIdx_")]
    lds_idx: i64,
    #[serde(rename = "dsName_", default)]
    ds_name: String,
    #[serde(rename = "dsType_", default)]
    ds_type: String,
    #[serde(rename = "segment_", default)]
    segment: Option<String>,
    #[serde(rename = "memOrg_", default)]
    mem_org: BTreeMap<String, WireMemOrgJson>,
}

/// `memOrg_`'s serde shape.
#[derive(Deserialize)]
struct WireMemOrgJson {
    #[serde(rename = "isPresent", default)]
    is_present: i64,
    #[serde(rename = "isPadded", default)]
    is_padded: i64,
    #[serde(rename = "isZeroPadded", default)]
    is_zero_padded: i64,
    #[serde(rename = "zpadGapFront", default)]
    zpad_gap_front: Vec<i64>,
    #[serde(rename = "gapPerDim", default)]
    gap_per_dim: BTreeMap<String, i64>,
    #[serde(rename = "dsOffset", default)]
    ds_offset: i64,
    #[serde(rename = "allocateNode_", default)]
    allocate_node: String,
}

/// The default the C++ leaves `target_` at where the dumper writes nothing.
fn default_target() -> String {
    "undefined".to_owned()
}
/// READ THE WHOLE FILE — the boundary.
///
/// # Errors
///
/// One [`Refusal`] per way the C++ importer refuses, with the field and the evidence.
pub fn read_file(text: &str) -> Result<WireFile, Refusal> {
    let top: WireTopLevel = serde_json::from_str(text).map_err(|e| Refusal::MalformedTopLevel {
        message: e.to_string(),
    })?;
    let mut programs = Vec::with_capacity(top.len());
    for (name, program) in top {
        let target = SenTarget::from_spelling(&program.target).ok_or(Refusal::UnknownSpelling {
            spelling: program.target.clone(),
            field: "target_",
        })?;
        let mut core_id_to_dsc = BTreeMap::new();
        for (core_str, arr_id) in program.core_id_to_dsc {
            let core: i64 = core_str
                .parse()
                .map_err(|_| Refusal::MalformedInteger {
                    text: core_str,
                    field: "coreIdToDsc_ key".into(),
                })?;
            // A negative array id is the C++'s nullptr entry (`dsc/superdsc.cpp:830-835`).
            core_id_to_dsc.insert(core, (arr_id >= 0).then_some(arr_id as usize));
        }
        let mut ops = Vec::with_capacity(program.dscs.len());
        for dsc in program.dscs {
            // A `dscs_` entry is a one-key map from the op's own name to its fields.
            for (op_name, op) in dsc {
                ops.push(read_op(&op_name, op)?);
            }
        }
        programs.push(WireProgram {
            name,
            num_cores_used: program.num_cores_used,
            core_id_to_dsc,
            target,
            ops,
        });
    }
    Ok(WireFile { programs })
}

/// READ ONE OP — scan 1 (names, linkage, validation), then scan 2 (per-kind fields).
fn read_op(name: &str, op: WireOpJson) -> Result<WireOp, Refusal> {
    // -- scan 1: collect names, refuse duplicates, resolve every prev_ -----------------------
    let mut by_name: BTreeMap<&str, usize> = BTreeMap::new();
    for (index, node) in op.schedule_tree.iter().enumerate() {
        let node_type = node.get("nodeType_").and_then(|v| v.as_str());
        let node_name = node.get("name_").and_then(|v| v.as_str());
        let prev = node.get("prev_").and_then(|v| v.as_str());
        if node_type.is_none() || node_name.is_none() || prev.is_none() {
            return Err(Refusal::MissingBasicFields { index });
        }
        let node_name = node_name.expect("checked above");
        if by_name.insert(node_name, index).is_some() {
            return Err(Refusal::DuplicateName {
                name: node_name.to_owned(),
            });
        }
    }
    for (index, node) in op.schedule_tree.iter().enumerate() {
        let prev = node.get("prev_").and_then(|v| v.as_str()).unwrap_or_default();
        let _ = index;
        if !prev.is_empty() && !by_name.contains_key(prev) {
            return Err(Refusal::UnknownParent {
                name: prev.to_owned(),
                child: node
                    .get("name_")
                    .and_then(|v| v.as_str())
                    .unwrap_or("<unnamed>")
                    .to_owned(),
            });
        }
    }

    // -- scan 2: per-kind fields --------------------------------------------------------------
    let mut tree = Vec::with_capacity(op.schedule_tree.len());
    for node in &op.schedule_tree {
        tree.push(read_node(node, &by_name)?);
    }

    // -- the allocate cross-checks ------------------------------------------------------------
    for node in &tree {
        let NodeKind::Allocate(alloc) = &node.kind else {
            continue;
        };
        for user in alloc.alloc_users.keys() {
            if !by_name.contains_key(user.as_str()) {
                return Err(Refusal::DanglingAllocUser {
                    user: user.clone(),
                    alloc: node.name.clone(),
                });
            }
        }
    }

    // -- labeledDs -----------------------------------------------------------------------------
    let mut labeled_ds = Vec::with_capacity(op.labeled_ds.len());
    for lds in &op.labeled_ds {
        let mut mem_org = BTreeMap::new();
        for (comp_str, entry) in &lds.mem_org {
            let comp =
                SenComponent::from_spelling(comp_str).ok_or(Refusal::UnknownSpelling {
                    spelling: comp_str.clone(),
                    field: "memOrg_",
                })?;
            let mut gap_per_dim = BTreeMap::new();
            for (dim_str, gap) in &entry.gap_per_dim {
                let dim = PrimaryDim::from_spelling(dim_str).ok_or(Refusal::UnknownDim {
                    spelling: dim_str.clone(),
                    field: "gapPerDim",
                })?;
                gap_per_dim.insert(dim, *gap);
            }
            mem_org.insert(
                comp,
                MemOrgEntry {
                    is_present: entry.is_present != 0,
                    is_padded: entry.is_padded != 0,
                    is_zero_padded: entry.is_zero_padded != 0,
                    zpad_gap_front: entry.zpad_gap_front.clone(),
                    gap_per_dim,
                    ds_offset: entry.ds_offset,
                    allocate_node: (!entry.allocate_node.is_empty())
                        .then(|| entry.allocate_node.clone()),
                },
            );
        }
        let segment = match lds.segment.as_deref() {
            None | Some("") => None,
            Some(text) => Some(LdsSegment::from_spelling(text).ok_or(
                Refusal::UnknownSpelling {
                    spelling: text.to_owned(),
                    field: "segment_",
                },
            )?),
        };
        labeled_ds.push(LabeledDs {
            lds_idx: lds.lds_idx,
            ds_name: lds.ds_name.clone(),
            ds_type: lds.ds_type.clone(),
            segment,
            mem_org,
        });
    }

    Ok(WireOp {
        name: name.to_owned(),
        num_cores_used: op.num_cores_used,
        num_corelets_used: op.num_corelets_used,
        core_ids_used: op.core_ids_used,
        num_corelets_used_dsc2: op.num_corelets_used_dsc2,
        schedule_tree: tree,
        labeled_ds,
    })
}

/// READ ONE NODE — scan 2's dispatch on the (already-validated) `nodeType_`.
fn read_node(
    node: &serde_json::Value,
    by_name: &BTreeMap<&str, usize>,
) -> Result<ScheduleNode, Refusal> {
    let kind_str = node.get("nodeType_").and_then(|v| v.as_str()).unwrap_or("");
    let node_type =
        enums::NodeType::from_spelling(kind_str).ok_or(Refusal::UnknownNodeType {
            spelling: kind_str.to_owned(),
        })?;
    let name = node
        .get("name_")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_owned();
    let prev = node
        .get("prev_")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_owned();
    let mut relevant = BTreeMap::new();
    if let Some(comps) = node.get("relevantComps_") {
        let wire: tree::WireRelevantComps = serde_json::from_value(comps.clone())
            .map_err(|e| Refusal::MalformedNode {
                message: format!("relevantComps_: {e}"),
            })?;
        relevant = tree::relevant_comps(wire)?;
    }

    // A node-name reference: empty string is the C++'s "no reference", anything else must resolve.
    let node_ref = |field: &'static str,
                    value: &serde_json::Value|
     -> Result<Option<usize>, Refusal> {
        let name = value.as_str().unwrap_or_default();
        if name.is_empty() || value.is_null() {
            return Ok(None);
        }
        Ok(by_name.get(name).copied().ok_or(Refusal::UnknownParent {
            name: name.to_owned(),
            child: field.to_owned(),
        })?
        .into())
    };

    let kind = match node_type {
        enums::NodeType::Block => NodeKind::Block,
        enums::NodeType::Loop => NodeKind::Loop(read_loop(node)?),
        enums::NodeType::Transfer => NodeKind::Transfer(read_transfer(node, &node_ref)?),
        enums::NodeType::Compute => NodeKind::Compute(read_compute(node, by_name)?),
        enums::NodeType::Sync => NodeKind::Sync(read_sync(node, &node_ref)?),
        enums::NodeType::Allocate => NodeKind::Allocate(read_allocate(node, &node_ref)?),
        enums::NodeType::Condition | enums::NodeType::StickMask => {
            // The importer has arms for both (`dsc/dsc2.cpp:1428-1432`, `:1851`+); no fixture
            // carries one yet, so the reader refuses rather than guessing the shape. This is the
            // refusal the next fixture that exercises the arm will surface as.
            return Err(Refusal::UnknownNodeType {
                spelling: kind_str.to_owned(),
            });
        }
    };

    Ok(ScheduleNode {
        name,
        prev,
        relevant_comps: relevant,
        kind,
    })
}

/// SCAN 2's LOOP arm (`dsc/dsc2.cpp:1403-1427`).
fn read_loop(node: &serde_json::Value) -> Result<LoopNode, Refusal> {
    let num_id = node
        .get("numId_")
        .map(|v| json_i64(v, "numId_"))
        .transpose()?
        .unwrap_or(0);
    let den_id = node
        .get("denId_")
        .map(|v| json_i64(v, "denId_"))
        .transpose()?
        .unwrap_or(1);
    let parametric = node
        .get("parametricLoop_")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        != 0;
    let parametric_lds_idx = node
        .get("parametricLdsIdx_")
        .map(|v| json_i64(v, "parametricLdsIdx_"))
        .transpose()?
        .unwrap_or(-1);
    let mut dims = Vec::new();
    if let Some(entries) = node.get("dims_").and_then(|v| v.as_array()) {
        for entry in entries {
            let dim_str = entry.get("dim_").and_then(|v| v.as_str()).unwrap_or("");
            let kind_str = entry.get("kind_").and_then(|v| v.as_str()).unwrap_or("");
            dims.push(LoopDim {
                dim: PrimaryDim::from_spelling(dim_str).ok_or(Refusal::UnknownDim {
                    spelling: dim_str.to_owned(),
                    field: "dims_",
                })?,
                kind: MetaDimKind::from_spelling(kind_str).ok_or(Refusal::UnknownSpelling {
                    spelling: kind_str.to_owned(),
                    field: "dims_ kind_",
                })?,
            });
        }
    }
    let mut loop_count_symbol_ids = BTreeMap::new();
    if let Some(syms) = node.get("loopCountSymbolIds_") {
        for (k, v) in syms.as_object().into_iter().flatten() {
            let key: i64 = k.parse().map_err(|_| Refusal::MalformedInteger {
                text: k.clone(),
                field: "loopCountSymbolIds_ key".into(),
            })?;
            loop_count_symbol_ids.insert(key, json_i64(v, "loopCountSymbolIds_")?);
        }
    }
    Ok(LoopNode {
        num_id,
        den_id,
        parametric,
        parametric_lds_idx,
        dims,
        loop_count_symbol_ids,
    })
}

/// A JSON value that may be a number or a string — `ImportUtil::importData`'s integral arm.
fn json_i64(value: &serde_json::Value, field: &'static str) -> Result<i64, Refusal> {
    match value {
        serde_json::Value::Number(n) => n
            .as_i64()
            .ok_or(Refusal::MalformedInteger {
                text: n.to_string(),
                field: field.into(),
            }),
        serde_json::Value::String(s) => s.parse().map_err(|_| Refusal::MalformedInteger {
            text: s.clone(),
            field: field.into(),
        }),
        _ => Err(Refusal::MalformedInteger {
            text: value.to_string(),
            field: field.into(),
        }),
    }
}

/// SCAN 2's TRANSFER arm (`dsc/dsc2.cpp:1444-1636`).
fn read_transfer(
    node: &serde_json::Value,
    node_ref: &dyn Fn(&'static str, &serde_json::Value) -> Result<Option<usize>, Refusal>,
) -> Result<TransferNode, Refusal> {
    let loc = |value: &serde_json::Value| -> Result<Loc, Refusal> {
        let wire: tree::WireLoc = serde_json::from_value(value.clone()).map_err(|e| {
            Refusal::MalformedNode {
                message: format!("src_/dstVias_: {e}"),
            }
        })?;
        wire.try_into()
    };
    let src = loc(
        node.get("src_")
            .ok_or(Refusal::UnknownSpelling {
                spelling: String::new(),
                field: "src_",
            })?,
    )?;
    let src_indirect = node.get("srcIndirect_").map(loc).transpose()?;
    let mut dst_vias = Vec::new();
    if let Some(vias) = node.get("dstVias_").and_then(|v| v.as_array()) {
        for via in vias {
            let loc_wire = via.get("loc_").ok_or(Refusal::MalformedNode {
                message: "dstVias_ entry without loc_".into(),
            })?;
            let via_loc = loc(loc_wire)?;
            let loc_indirect = via.get("locIndirect_").map(loc).transpose()?;
            let mut route = Vec::new();
            for hop in via.get("via_").and_then(|v| v.as_array()).into_iter().flatten() {
                let hop_str = hop.as_str().unwrap_or_default();
                route.push(SenComponent::from_spelling(hop_str).ok_or(
                    Refusal::UnknownSpelling {
                        spelling: hop_str.to_owned(),
                        field: "via_",
                    },
                )?);
            }
            dst_vias.push(DstVia {
                loc: via_loc,
                loc_indirect,
                via: route,
            });
        }
    }
    let last_fusable_parent_loop_src = node
        .get("lastFusableParentLoopSrc_")
        .and_then(|v| node_ref("lastFusableParentLoopSrc_", v).ok())
        .flatten();
    let mut last_fusable_parent_loop_dst = Vec::new();
    if let Some(dsts) = node.get("lastFusableParentLoopDst_").and_then(|v| v.as_array()) {
        for dst in dsts {
            last_fusable_parent_loop_dst.push(node_ref("lastFusableParentLoopDst_", dst)?);
        }
    }
    let data_info = |value: &serde_json::Value| -> Result<DataInfo, Refusal> {
        read_data_info(value, node_ref)
    };
    let src_lds_and_loop_offsets = node
        .get("srcLdsAndLoopOffsets_")
        .map(data_info)
        .transpose()?
        .unwrap_or_else(empty_data_info);
    let src_indirect_lds_and_loop_offsets = node
        .get("srcIndirectLdsAndLoopOffsets_")
        .map(data_info)
        .transpose()?;
    let mut dst_lds_and_loop_offsets = Vec::new();
    if let Some(dsts) = node.get("dstLdsAndLoopOffsets_").and_then(|v| v.as_array()) {
        for dst in dsts {
            dst_lds_and_loop_offsets.push(data_info(dst)?);
        }
    }
    let dst_indirect_lds_and_loop_offsets = node
        .get("dstIndirectLdsAndLoopOffsets_")
        .map(|v| -> Result<Vec<DataInfo>, Refusal> {
            v.as_array()
                .ok_or(Refusal::MalformedNode {
                    message: "dstIndirectLdsAndLoopOffsets_ is not an array".into(),
                })?
                .iter()
                .map(&data_info)
                .collect()
        })
        .transpose()?;
    let chunk = |entry: &serde_json::Value| -> Result<ChunkSize, Refusal> {
        Ok(ChunkSize {
            size_dim: read_size(
                entry.get("sizeDim_").ok_or(Refusal::MalformedNode {
                    message: "chunk entry without sizeDim_".into(),
                })?,
            )?,
            src_size_idx: json_i64(
                entry.get("srcSizeIdx_").unwrap_or(&serde_json::Value::Null),
                "srcSizeIdx_",
            )
            .unwrap_or(-1),
            dst_size_idx: json_i64(
                entry.get("dstSizeIdx_").unwrap_or(&serde_json::Value::Null),
                "dstSizeIdx_",
            )
            .unwrap_or(-1),
        })
    };
    let mut chunk_size = Vec::new();
    for entry in node
        .get("unitTimeTransferChunkSize_")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
    {
        chunk_size.push(chunk(entry)?);
    }
    let mut chunk_stride = Vec::new();
    for entry in node
        .get("unitTimeTransferChunkStride_")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
    {
        chunk_stride.push(chunk(entry)?);
    }
    let mut core_id_to_gtr_info = BTreeMap::new();
    if let Some(gtrs) = node.get("coreIdToGTRInfo_").and_then(|v| v.as_object()) {
        for (core_str, info) in gtrs {
            let core: i64 = core_str.parse().map_err(|_| Refusal::MalformedInteger {
                text: core_str.clone(),
                field: "coreIdToGTRInfo_ key".into(),
            })?;
            core_id_to_gtr_info.insert(
                core,
                GtrInfo {
                    group_id: json_i64(
                        info.get("groupId_").unwrap_or(&serde_json::Value::Null),
                        "groupId_",
                    )
                    .unwrap_or(-1),
                    num_sharers: json_i64(
                        info.get("numSharers_").unwrap_or(&serde_json::Value::Null),
                        "numSharers_",
                    )
                    .unwrap_or(-1),
                },
            );
        }
    }
    let mut transfer_size = BTreeMap::new();
    if let Some(sizes) = node.get("transferSize_").and_then(|v| v.as_object()) {
        for (dim_str, value) in sizes {
            let dim = PrimaryDim::from_spelling(dim_str).ok_or(Refusal::UnknownDim {
                spelling: dim_str.clone(),
                field: "transferSize_",
            })?;
            transfer_size.insert(dim, json_i64(value, "transferSize_")?);
        }
    }
    let mut corelet_views = BTreeMap::new();
    if let Some(views) = node.get("coreletViews_").and_then(|v| v.as_object()) {
        for (cl_str, view) in views {
            let cl: i64 = cl_str.parse().map_err(|_| Refusal::MalformedInteger {
                text: cl_str.clone(),
                field: "coreletViews_ key".into(),
            })?;
            corelet_views.insert(cl, read_corelet_view(view, node_ref)?);
        }
    }
    let coords = node
        .get("coordinates_")
        .map(coords::coordinates)
        .transpose()?
        .flatten();
    Ok(TransferNode {
        src,
        src_indirect,
        dst_vias,
        last_fusable_parent_loop_src,
        last_fusable_parent_loop_dst,
        src_lds_and_loop_offsets,
        src_indirect_lds_and_loop_offsets,
        dst_lds_and_loop_offsets,
        dst_indirect_lds_and_loop_offsets,
        replication_factor: node
            .get("replicationFactor_")
            .map(|v| json_i64(v, "replicationFactor_"))
            .transpose()?
            .unwrap_or(1),
        chunk_size,
        chunk_stride,
        num_chunks: node
            .get("unitTimeTransferNumChunks_")
            .map(|v| json_i64(v, "unitTimeTransferNumChunks_"))
            .transpose()?
            .unwrap_or(1),
        rotate_num_elements: node
            .get("rotateNumElements_")
            .map(|v| json_i64(v, "rotateNumElements_"))
            .transpose()?
            .unwrap_or(0),
        core_id_to_gtr_info,
        transfer_size,
        corelet_views,
        coordinates: coords,
    })
}

/// ONE `coreletViews_` ENTRY of a transfer (`dsc/dsc2.cpp:1563-1600`).
fn read_corelet_view(
    view: &serde_json::Value,
    node_ref: &dyn Fn(&'static str, &serde_json::Value) -> Result<Option<usize>, Refusal>,
) -> Result<CoreletView, Refusal> {
    let unit_view = |value: &serde_json::Value| -> Result<UnitView, Refusal> {
        read_unit_view(value, node_ref)
    };
    let one = |field: &str| -> Result<UnitView, Refusal> {
        view.get(field)
            .map(unit_view)
            .transpose()
            .map(|v| v.unwrap_or_default())
    };
    let many = |field: &str| -> Result<Vec<UnitView>, Refusal> {
        view.get(field)
            .and_then(|v| v.as_array())
            .map(|entries| entries.iter().map(unit_view).collect())
            .transpose()
            .map(|v| v.unwrap_or_default())
    };
    Ok(CoreletView {
        src_loops_and_size: one("srcLoopsAndSize_")?,
        src_indirect_loops_and_size: one("srcIndirectLoopsAndSize_")?,
        dst_loops_and_sizes: many("dstLoopsAndSizes_")?,
        dst_indirect_loops_and_sizes: many("dstIndirectLoopsAndSizes_")?,
    })
}

/// ONE `importUnitView` (`dsc/dsc2.cpp:1252-1265`).
fn read_unit_view(
    view: &serde_json::Value,
    node_ref: &dyn Fn(&'static str, &serde_json::Value) -> Result<Option<usize>, Refusal>,
) -> Result<UnitView, Refusal> {
    let mut sizes_no_gaps = Vec::new();
    for size in view
        .get("sizesNoGaps_")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
    {
        sizes_no_gaps.push(read_size(size)?);
    }
    let loop_info = |entry: &serde_json::Value| -> Result<LoopInfo, Refusal> {
        let loop_name = entry.get("loop_").and_then(|v| v.as_str()).unwrap_or("");
        let loop_idx = node_ref("loop_", entry.get("loop_").unwrap_or(&serde_json::Value::Null))?
            .ok_or(Refusal::UnknownParent {
                name: loop_name.to_owned(),
                child: "loop_".into(),
            })?;
        let dim_str = entry.get("dim_").and_then(|v| v.as_str()).unwrap_or("");
        Ok(LoopInfo {
            loop_idx,
            dim: PrimaryDim::from_spelling(dim_str).ok_or(Refusal::UnknownDim {
                spelling: dim_str.to_owned(),
                field: "loop_ dim_",
            })?,
            size_idx: json_i64(
                entry.get("sizeIdx_").unwrap_or(&serde_json::Value::Null),
                "sizeIdx_",
            )
            .unwrap_or(-1),
            elem_offset: json_i64(
                entry.get("elemOffset_").unwrap_or(&serde_json::Value::Null),
                "elemOffset_",
            )
            .unwrap_or(0),
        })
    };
    let mut composite_loops = Vec::new();
    for entry in view
        .get("compositeLoops_")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
    {
        composite_loops.push(loop_info(entry)?);
    }
    let mut outer_loops = Vec::new();
    for entry in view
        .get("outerLoops_")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
    {
        outer_loops.push(loop_info(entry)?);
    }
    let mut sizes_with_gaps = BTreeMap::new();
    if let Some(gaps) = view.get("sizesWithGaps_").and_then(|v| v.as_object()) {
        for (core_str, sizes) in gaps {
            let core: i64 = core_str.parse().map_err(|_| Refusal::MalformedInteger {
                text: core_str.clone(),
                field: "sizesWithGaps_ key".into(),
            })?;
            let mut per_core = Vec::new();
            for size in sizes.as_array().into_iter().flatten() {
                per_core.push(read_size(size)?);
            }
            sizes_with_gaps.insert(core, per_core);
        }
    }
    Ok(UnitView {
        sizes_no_gaps,
        composite_loops,
        outer_loops,
        sizes_with_gaps,
    })
}

/// ONE `Size` (`dsc/dsc2.cpp:1214-1217`).
fn read_size(size: &serde_json::Value) -> Result<Size, Refusal> {
    let dim_str = size.get("dim_").and_then(|v| v.as_str()).unwrap_or("");
    Ok(Size {
        dim: PrimaryDim::from_spelling(dim_str).ok_or(Refusal::UnknownDim {
            spelling: dim_str.to_owned(),
            field: "sizesNoGaps_ dim_",
        })?,
        size: json_i64(
            size.get("size_").unwrap_or(&serde_json::Value::Null),
            "size_",
        )
        .unwrap_or(0),
    })
}

/// ONE `DataInfo` (`dsc/dsc2.cpp:1218-1250`, `importDataInfo`).
fn read_data_info(
    info: &serde_json::Value,
    node_ref: &dyn Fn(&'static str, &serde_json::Value) -> Result<Option<usize>, Refusal>,
) -> Result<DataInfo, Refusal> {
    let start_addr = info
        .get("startAddr_")
        .map(fold::fold_data)
        .transpose()?
        .flatten()
        .unwrap_or_else(|| FoldData::ZeroFold("0".to_owned()));
    let mut const_ele_offsets = BTreeMap::new();
    if let Some(cores) = info.get("constEleOffsets_").and_then(|v| v.as_object()) {
        for (core_str, corelets) in cores {
            let core: i64 = core_str.parse().map_err(|_| Refusal::MalformedInteger {
                text: core_str.clone(),
                field: "constEleOffsets_ key".into(),
            })?;
            let mut per_corelet = BTreeMap::new();
            for (cl_str, dims) in corelets.as_object().into_iter().flatten() {
                let cl: i64 = cl_str.parse().map_err(|_| Refusal::MalformedInteger {
                    text: cl_str.clone(),
                    field: "constEleOffsets_ corelet".into(),
                })?;
                let mut per_dim = BTreeMap::new();
                for (dim_str, offset) in dims.as_object().into_iter().flatten() {
                    let dim = PrimaryDim::from_spelling(dim_str).ok_or(Refusal::UnknownDim {
                        spelling: dim_str.clone(),
                        field: "constEleOffsets_",
                    })?;
                    per_dim.insert(dim, json_i64(offset, "constEleOffsets_")?);
                }
                per_corelet.insert(cl, per_dim);
            }
            const_ele_offsets.insert(core, per_corelet);
        }
    }
    let mut loop_ele_offsets = BTreeMap::new();
    if let Some(corelets) = info.get("loopEleOffsets_").and_then(|v| v.as_object()) {
        for (cl_str, loops) in corelets {
            let cl: i64 = cl_str.parse().map_err(|_| Refusal::MalformedInteger {
                text: cl_str.clone(),
                field: "loopEleOffsets_ key".into(),
            })?;
            let mut per_loop = BTreeMap::new();
            for (loop_str, dims) in loops.as_object().into_iter().flatten() {
                let mut per_dim = BTreeMap::new();
                for (dim_str, offset) in dims.as_object().into_iter().flatten() {
                    let dim = PrimaryDim::from_spelling(dim_str).ok_or(Refusal::UnknownDim {
                        spelling: dim_str.clone(),
                        field: "loopEleOffsets_",
                    })?;
                    per_dim.insert(dim, json_i64(offset, "loopEleOffsets_")?);
                }
                per_loop.insert(loop_str.clone(), per_dim);
            }
            loop_ele_offsets.insert(cl, per_loop);
        }
    }
    let mut buffer_addr_offset = BTreeMap::new();
    if let Some(cores) = info.get("bufferAddrOffset_").and_then(|v| v.as_object()) {
        for (core_str, corelets) in cores {
            let core: i64 = core_str.parse().map_err(|_| Refusal::MalformedInteger {
                text: core_str.clone(),
                field: "bufferAddrOffset_ key".into(),
            })?;
            let mut per_corelet = BTreeMap::new();
            for (cl_str, offset) in corelets.as_object().into_iter().flatten() {
                let cl: i64 = cl_str.parse().map_err(|_| Refusal::MalformedInteger {
                    text: cl_str.clone(),
                    field: "bufferAddrOffset_ corelet".into(),
                })?;
                per_corelet.insert(cl, json_i64(offset, "bufferAddrOffset_")?);
            }
            buffer_addr_offset.insert(core, per_corelet);
        }
    }
    let buffer_switch_position = node_ref(
        "bufferSwitchPosition_",
        info.get("bufferSwitchPosition_")
            .unwrap_or(&serde_json::Value::Null),
    )?;
    Ok(DataInfo {
        my_lds_idx: json_i64(
            info.get("myLdsIdx_").unwrap_or(&serde_json::Value::Null),
            "myLdsIdx_",
        )
        .unwrap_or(-1),
        start_addr,
        is_start_addr_symbolic: info
            .get("isStartAddrSymbolic_")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            != 0,
        latch_data_id: json_i64(
            info.get("latchDataId_").unwrap_or(&serde_json::Value::Null),
            "latchDataId_",
        )
        .unwrap_or(-1),
        constant_id: json_i64(
            info.get("constantId_").unwrap_or(&serde_json::Value::Null),
            "constantId_",
        )
        .unwrap_or(-1),
        const_ele_offsets,
        loop_ele_offsets,
        buffer_addr_offset,
        buffer_switch_position,
        data_connect: info
            .get("dataConnect_")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| DataConnect(s.to_owned())),
    })
}

/// A DataInfo with every field at its C++ default — for a `srcLdsAndLoopOffsets_` the dumper
/// omitted.
fn empty_data_info() -> DataInfo {
    DataInfo {
        my_lds_idx: -1,
        start_addr: FoldData::ZeroFold("0".to_owned()),
        is_start_addr_symbolic: false,
        latch_data_id: -1,
        constant_id: -1,
        const_ele_offsets: BTreeMap::new(),
        loop_ele_offsets: BTreeMap::new(),
        buffer_addr_offset: BTreeMap::new(),
        buffer_switch_position: None,
        data_connect: None,
    }
}

/// SCAN 2's COMPUTE arm (`dsc/dsc2.cpp:1638-1714`).
fn read_compute(node: &serde_json::Value, by_name: &BTreeMap<&str, usize>) -> Result<ComputeNode, Refusal> {
    let ex_unit_str = node.get("exUnit_").and_then(|v| v.as_str()).unwrap_or("");
    let type_str = node.get("type_").and_then(|v| v.as_str()).unwrap_or("");
    let format_str = node.get("dataFormat_").and_then(|v| v.as_str()).unwrap_or("");
    let units = |field: &'static str| -> Result<Vec<SenComponent>, Refusal> {
        let mut out = Vec::new();
        for unit in node
            .get(field)
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
        {
            let unit_str = unit.as_str().unwrap_or_default();
            out.push(SenComponent::from_spelling(unit_str).ok_or(
                Refusal::UnknownSpelling {
                    spelling: unit_str.to_owned(),
                    field,
                },
            )?);
        }
        Ok(out)
    };
    let data_infos = |field: &'static str,
                      node_ref: &dyn Fn(
        &'static str,
        &serde_json::Value,
    ) -> Result<Option<usize>, Refusal>|
     -> Result<Vec<DataInfo>, Refusal> {
        let mut out = Vec::new();
        for info in node.get(field).and_then(|v| v.as_array()).into_iter().flatten() {
            out.push(read_data_info(info, node_ref)?);
        }
        Ok(out)
    };
    // The compute arm needs node refs only for `bufferSwitchPosition_` inside its DataInfos and
    // the loop names inside its corelet views; the tree is complete by now, so resolve against it.
    let node_ref = |field: &'static str,
                    value: &serde_json::Value|
     -> Result<Option<usize>, Refusal> {
        let name = value.as_str().unwrap_or_default();
        if name.is_empty() || value.is_null() {
            return Ok(None);
        }
        Ok(by_name.get(name).copied().ok_or(Refusal::UnknownParent {
            name: name.to_owned(),
            child: field.to_owned(),
        })?
        .into())
    };
    let mut corelet_views = BTreeMap::new();
    if let Some(views) = node.get("coreletViews_").and_then(|v| v.as_object()) {
        for (cl_str, view) in views {
            let cl: i64 = cl_str.parse().map_err(|_| Refusal::MalformedInteger {
                text: cl_str.clone(),
                field: "coreletViews_ key".into(),
            })?;
            let unit_view = |value: &serde_json::Value| -> Result<UnitView, Refusal> {
                read_unit_view(value, &node_ref)
            };
            let many =
                |field: &str| -> Result<Vec<UnitView>, Refusal> {
                    view.get(field)
                        .and_then(|v| v.as_array())
                        .map(|entries| entries.iter().map(unit_view).collect())
                        .transpose()
                        .map(|v| v.unwrap_or_default())
                };
            corelet_views.insert(
                cl,
                ComputeCoreletView {
                    inputs_loops_and_sizes: many("inputsLoopsAndSizes_")?,
                    outputs_loops_and_sizes: many("outputsLoopsAndSizes_")?,
                },
            );
        }
    }
    let mut input_coordinates = BTreeMap::new();
    if let Some(coords) = node.get("inputCoordinates_").and_then(|v| v.as_object()) {
        for (pos_str, coord) in coords {
            let pos: i64 = pos_str.parse().map_err(|_| Refusal::MalformedInteger {
                text: pos_str.clone(),
                field: "inputCoordinates_ key".into(),
            })?;
            // The dumper wraps each entry in a `coordinates_` key
            // (`dsc/dsc2.cpp:1704`), while a node's own `coordinates_` field is the
            // object itself (`:1632`).
            let inner = coord.get("coordinates_").unwrap_or(coord);
            input_coordinates.insert(
                pos,
                coords::coordinates(inner)?
                    .ok_or(Refusal::MalformedCoordinates {
                        message: "null inputCoordinates_ entry".into(),
                    })?,
            );
        }
    }
    Ok(ComputeNode {
        ex_unit: SenComponent::from_spelling(ex_unit_str).ok_or(Refusal::UnknownSpelling {
            spelling: ex_unit_str.to_owned(),
            field: "exUnit_",
        })?,
        ty: WireComputeType::from_spelling(type_str).ok_or(Refusal::UnknownSpelling {
            spelling: type_str.to_owned(),
            field: "type_",
        })?,
        data_format: enums::data_format(format_str).ok_or(Refusal::UnknownSpelling {
            spelling: format_str.to_owned(),
            field: "dataFormat_",
        })?,
        inputs: units("inputs_")?,
        outputs: units("outputs_")?,
        inputs_lds_and_loop_offsets: data_infos("inputsLdsAndLoopOffsets_", &node_ref)?,
        outputs_lds_and_loop_offsets: data_infos("outputsLdsAndLoopOffsets_", &node_ref)?,
        corelet_views,
        num_folds_engaged: node
            .get("numFoldsEngaged")
            .map(|v| json_i64(v, "numFoldsEngaged"))
            .transpose()?
            .unwrap_or(1),
        is_opaque_op: node
            .get("isOpaqueOp_")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            != 0,
        instr_attribute: node
            .get("instrAttribute_")
            .and_then(|v| v.as_object())
            .map(|attrs| {
                attrs
                    .iter()
                    .map(|(k, v)| (k.to_owned(), v.to_string()))
                    .collect()
            })
            .unwrap_or_default(),
        input_coordinates,
        output_coordinates: node
            .get("outputCoordinates_")
            .map(|v| {
                // The dumper wraps this one in a `coordinates_` key too (`dsc/dsc2.cpp:1709`).
                coords::coordinates(v.get("coordinates_").unwrap_or(v))
            })
            .transpose()?
            .flatten(),
    })
}

/// SCAN 2's SYNC arm (`dsc/dsc2.cpp:1716-1750`).
fn read_sync(
    node: &serde_json::Value,
    node_ref: &dyn Fn(&'static str, &serde_json::Value) -> Result<Option<usize>, Refusal>,
) -> Result<SyncNode, Refusal> {
    let mut units = Vec::new();
    for unit in node
        .get("units_")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
    {
        let unit_str = unit.as_str().unwrap_or_default();
        units.push(SenComponent::from_spelling(unit_str).ok_or(Refusal::UnknownSpelling {
            spelling: unit_str.to_owned(),
            field: "units_",
        })?);
    }
    let mut other_end = Vec::new();
    for other in node
        .get("otherEndOfTheSignals_")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
    {
        other_end.push(
            node_ref("otherEndOfTheSignals_", other)?
                .ok_or(Refusal::UnknownParent {
                    name: other.as_str().unwrap_or_default().to_owned(),
                    child: "otherEndOfTheSignals_".into(),
                })?,
        );
    }
    Ok(SyncNode {
        units,
        is_receive: node
            .get("isReceive_")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            != 0,
        is_soft: node
            .get("isSoft_")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            != 0,
        implicit_sync_ref_transfer: node_ref(
            "implicitSyncRefTransfer_",
            node.get("implicitSyncRefTransfer_")
                .unwrap_or(&serde_json::Value::Null),
        )?,
        other_end_of_the_signals: other_end,
    })
}

/// SCAN 2's ALLOCATE arm (`dsc/dsc2.cpp:1752-1850`).
fn read_allocate(
    node: &serde_json::Value,
    node_ref: &dyn Fn(&'static str, &serde_json::Value) -> Result<Option<usize>, Refusal>,
) -> Result<AllocateNode, Refusal> {
    let component_str = node.get("component_").and_then(|v| v.as_str()).unwrap_or("");
    let mut padding = BTreeMap::new();
    if let Some(pads) = node.get("padding_").and_then(|v| v.as_object()) {
        for (dim_str, pad_str) in pads {
            let dim = PrimaryDim::from_spelling(dim_str).ok_or(Refusal::UnknownDim {
                spelling: dim_str.clone(),
                field: "padding_",
            })?;
            let pad = pad_str.as_str().unwrap_or_default();
            padding.insert(
                dim,
                enums::PadType::from_spelling(pad).ok_or(Refusal::UnknownSpelling {
                    spelling: pad.to_owned(),
                    field: "padding_",
                })?,
            );
        }
    }
    let mut layout_dim_order = Vec::new();
    for dim in node
        .get("layoutDimOrder_")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
    {
        let dim_str = dim.as_str().unwrap_or_default();
        layout_dim_order.push(PrimaryDim::from_spelling(dim_str).ok_or(Refusal::UnknownDim {
            spelling: dim_str.to_owned(),
            field: "layoutDimOrder_",
        })?);
    }
    let mut max_dim_sizes = Vec::new();
    for size in node
        .get("maxDimSizes_")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
    {
        max_dim_sizes.push(json_i64(size, "maxDimSizes_")?);
    }
    let mut buffer_offset_core_corelet = BTreeMap::new();
    if let Some(cores) = node.get("bufferOffsetCoreCorelet_").and_then(|v| v.as_object()) {
        for (core_str, corelets) in cores {
            let core: i64 = core_str.parse().map_err(|_| Refusal::MalformedInteger {
                text: core_str.clone(),
                field: "bufferOffsetCoreCorelet_ key".into(),
            })?;
            let mut per_corelet = BTreeMap::new();
            for (cl_str, offset) in corelets.as_object().into_iter().flatten() {
                let cl: i64 = cl_str.parse().map_err(|_| Refusal::MalformedInteger {
                    text: cl_str.clone(),
                    field: "bufferOffsetCoreCorelet_ corelet".into(),
                })?;
                per_corelet.insert(cl, json_i64(offset, "bufferOffsetCoreCorelet_")?);
            }
            buffer_offset_core_corelet.insert(core, per_corelet);
        }
    }
    let mut back_gap_core = BTreeMap::new();
    if let Some(dims) = node.get("backGapCore_").and_then(|v| v.as_object()) {
        for (dim_str, cores) in dims {
            let dim = PrimaryDim::from_spelling(dim_str).ok_or(Refusal::UnknownDim {
                spelling: dim_str.clone(),
                field: "backGapCore_",
            })?;
            let mut per_core = BTreeMap::new();
            for (core_str, gap) in cores.as_object().into_iter().flatten() {
                let core: i64 = core_str.parse().map_err(|_| Refusal::MalformedInteger {
                    text: core_str.clone(),
                    field: "backGapCore_ core".into(),
                })?;
                per_core.insert(core, json_i64(gap, "backGapCore_")?);
            }
            back_gap_core.insert(dim, per_core);
        }
    }
    let mut gap_stick_spread = BTreeMap::new();
    if let Some(dims) = node.get("gapStickSpread_").and_then(|v| v.as_object()) {
        for (dim_str, spread) in dims {
            let dim = PrimaryDim::from_spelling(dim_str).ok_or(Refusal::UnknownDim {
                spelling: dim_str.clone(),
                field: "gapStickSpread_",
            })?;
            gap_stick_spread.insert(dim, json_i64(spread, "gapStickSpread_")?);
        }
    }
    let mut alloc_users = BTreeMap::new();
    if let Some(users) = node.get("allocUsers_").and_then(|v| v.as_object()) {
        for (user, count) in users {
            alloc_users.insert(
                user.clone(),
                json_i64(count, "allocUsers_")?,
            );
        }
    }
    Ok(AllocateNode {
        lds_idx: json_i64(
            node.get("ldsIdx_").unwrap_or(&serde_json::Value::Null),
            "ldsIdx_",
        )
        .unwrap_or(-1),
        const_idx: json_i64(
            node.get("constIdx_").unwrap_or(&serde_json::Value::Null),
            "constIdx_",
        )
        .unwrap_or(-1),
        temp_storage_for_compute: node_ref(
            "tempStorageForCompute_",
            node.get("tempStorageForCompute_")
                .unwrap_or(&serde_json::Value::Null),
        )?,
        component: SenComponent::from_spelling(component_str).ok_or(Refusal::UnknownSpelling {
            spelling: component_str.to_owned(),
            field: "component_",
        })?,
        padding,
        layout_dim_order,
        max_dim_sizes,
        num_buffers: json_i64(
            node.get("numBuffers_").unwrap_or(&serde_json::Value::Null),
            "numBuffers_",
        )
        .unwrap_or(1),
        start_address_core_corelet: node
            .get("startAddressCoreCorelet_")
            .map(fold::fold_data)
            .transpose()?
            .flatten()
            .unwrap_or_else(|| FoldData::ZeroFold("0".to_owned())),
        is_start_addr_symbolic: node
            .get("isStartAddrSymbolic_")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            != 0,
        buffer_offset_core_corelet,
        back_gap_core,
        indirect_alloc_type: {
            let text = node
                .get("indirectAllocType_")
                .and_then(|v| v.as_str())
                .unwrap_or("no_indirection");
            enums::IndirectAllocType::from_spelling(text).ok_or(Refusal::UnknownSpelling {
                spelling: text.to_owned(),
                field: "indirectAllocType_",
            })?
        },
        index_tensor_type: node
            .get("indexTensorType_")
            .and_then(|v| v.as_str())
            .map(str::to_owned),
        related_indirect_access_alloc: node_ref(
            "relatedIndirectAccessAlloc_",
            node.get("relatedIndirectAccessAlloc_")
                .unwrap_or(&serde_json::Value::Null),
        )?,
        ignore_symbolic_volume_limits: node
            .get("ignoreSymbolicVolumeLimits_")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            != 0,
        non_unified_alloc_in_hbm: node
            .get("nonUnifiedAllocInHBM_")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            != 0,
        gap_stick_spread,
        alloc_users,
        coordinates: node
            .get("coordinates_")
            .map(coords::coordinates)
            .transpose()?
            .flatten(),
    })
}
