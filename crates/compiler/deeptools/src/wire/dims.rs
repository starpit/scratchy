//! THE OP-LEVEL FIELDS THE LOWERING READS — `dataStageParam_`, `constantInfo_`, `computeOp_`.
//!
//! The C++ imports these three from the same dump the tree reader covers
//! (`dsc/dsc2.cpp:1122-1151`), except `computeOp_`, whose reader lives in the design-space
//! importer (`dsc/designSpaceConfig.cpp:7308-7370`) because the dsc2 import has no arm for it —
//! the SuperDSC it builds keeps the `computeOp_` the scheduler handed it rather than re-reading
//! one off the wire. The dumper writes all three for every op (`designSpaceConfig.cpp:6600+`), so
//! the reader parses all three and lets the adapter decide which to consult.
//!
//! # THE CONTRACT
//!
//! - [`DataStructDims`]'s reader transcribes `DataStructDims::importJsonObj`
//!   (`dsc/dims.cpp:311-435`) key for key: an unknown TOP-LEVEL key is the `DT_CHECK(0)`
//!   refusal (`:429-431`), while unknown keys INSIDE a `paddingSizes_` entry are silently
//!   ignored (that loop's else-less chain, `:408-427`) — which is why the dumper's own
//!   `totalSize_` (`dims.cpp:294`) is dropped, not refused.
//! - `constantInfo_` reads only the four keys the C++ reads (`dsc/dsc2.cpp:1138-1149`):
//!   `dataFormat_`, `name_`, `data_`, `isDataSymbolic_`. ⛔ `allocations_` IS DUMP-ONLY — the
//!   import ignores it and rebuilds the map from the allocate nodes' `constIdx_`
//!   (`dsc/dsc2.cpp:1836-1837`), so the reader does not parse it either.
//! - `computeOp_` carries the fields the adapter's consumers read: `superdsc.cpp:1343-1344`'s
//!   `computeOp_.at(0).opFuncName`, the padval default hunt over `opConsts` (`dsc2.cpp:5242`),
//!   and the linearization (`superdsc.cpp:1569-1596`: exUnit, opFuncName, attributes_, location,
//!   level, the labeledDs lists).

use std::collections::BTreeMap;

use sys_arch_spec::arch_enums::SenComponent;

use super::enums::{self, Fidelity, WireLoopName, WireOpFunc};
use super::fold::{fold_int_array, FoldedIntArray};
use super::{json_i64, Refusal};
use crate::generated::DataType;

/// ONE STAGE'S DIMENSIONS — `DataStructDims` (`dsc/dims.h:158-240`), the `ss_`/`el_` of a
/// `dataStageParam_` entry.
///
/// ⭐ THE PLAIN MEMBERS ARE `double` IN THE C++ (`dims.h:163-204`). The dumper prints them with
/// `operator<<` (`dims.cpp:189-195`), so a `0.5` stride survives to the wire; the value chain
/// (`primaryDimToVal_st`) multiplies and divides them. The reader keeps `f64` for those and
/// `i64` for the genuinely integral sub-structs, the way the C++ splits them.
#[derive(Debug, Clone, PartialEq)]
pub struct DataStructDims {
    /// `name_` — the stage name (`"core"`, `"chunk"`, `"2"`, `"2el"` …).
    pub name: String,
    /// `in_`.
    pub in_: f64,
    /// `out_`.
    pub out_: f64,
    /// `mb_`.
    pub mb_: f64,
    /// `ij_`.
    pub ij_: f64,
    /// `rc_`.
    pub rc_: f64,
    /// `kij_`.
    pub kij_: f64,
    /// `y_`.
    pub y_: f64,
    /// `x_`.
    pub x_: f64,
    /// `x1_`.
    pub x1_: f64,
    /// `sij_`.
    pub sij_: f64,
    /// `zij_`.
    pub zij_: f64,
    /// `i_`.
    pub i_: f64,
    /// `j_`.
    pub j_: f64,
    /// `r_`.
    pub r_: f64,
    /// `c_`.
    pub c_: f64,
    /// `ki_`.
    pub ki_: f64,
    /// `kj_`.
    pub kj_: f64,
    /// `si_`.
    pub si_: f64,
    /// `sj_`.
    pub sj_: f64,
    /// `zi_`.
    pub zi_: f64,
    /// `zj_`.
    pub zj_: f64,
    /// `symbolicDimInfo_` — dim → its max and granularity (`SymbolicDimInfo`, `dims.h:148-157`).
    pub symbolic_dim_info: BTreeMap<PrimaryDim, SymbolicDimInfo>,
    /// `maxSymbolicVolume_` — a set of dims, printed as the stringified key, → the volume cap.
    pub max_symbolic_volume: BTreeMap<String, i64>,
    /// `coreletSplit_` — dim → the work per corelet.
    pub corelet_split: BTreeMap<PrimaryDim, Vec<f64>>,
    /// `rowSplit_` — dim → corelet → the work per PT row.
    pub row_split: BTreeMap<PrimaryDim, BTreeMap<i64, Vec<f64>>>,
    /// `peSfpSplit_` — dim → corelet → component → the work for that unit.
    pub pe_sfp_split: BTreeMap<PrimaryDim, BTreeMap<i64, BTreeMap<SenComponent, f64>>>,
    /// `paddingSizes_` — dim → its padding facts.
    pub padding_sizes: BTreeMap<PrimaryDim, DimPaddingSizes>,
}

/// A symbolic dim's bounds — `SymbolicDimInfo` (`dsc/dims.h:148-157`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SymbolicDimInfo {
    /// `maxSize_`.
    pub max_size: i64,
    /// `granularity_`.
    pub granularity: i64,
}

/// A dim's padding facts — `DimPaddingSizes` (`dsc/dims.h:134-146`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimPaddingSizes {
    /// `padFront_`.
    pub pad_front: i64,
    /// `padBack_`.
    pub pad_back: i64,
    /// `unneededPad_` — total unneeded elements.
    pub unneeded_pad: i64,
    /// `unneededPadFront_`.
    pub unneeded_pad_front: i64,
    /// `unneededPadBack_`.
    pub unneeded_pad_back: i64,
    /// `stride_`.
    pub stride: i64,
    /// `dilation_`.
    pub dilation: i64,
    /// `windowDim_` — `PrimaryDimTypesCount` (no window) is `None` on the wire ("undefined").
    pub window_dim: Option<PrimaryDim>,
}

impl Default for DimPaddingSizes {
    fn default() -> Self {
        Self {
            pad_front: 0,
            pad_back: 0,
            unneeded_pad: 0,
            unneeded_pad_front: 0,
            unneeded_pad_back: 0,
            stride: 1,
            dilation: 1,
            window_dim: None,
        }
    }
}

/// ONE `dataStageParam_` ENTRY — `DataStage` (`dsc/dsc2.h:38-42`): the steady-state and
/// epilogue views of one stage, whose name is `ss_`'s.
#[derive(Debug, Clone, PartialEq)]
pub struct DataStage {
    /// `ss_` — steady state.
    pub ss: DataStructDims,
    /// `el_` — epilogue.
    pub el: DataStructDims,
}

impl DataStage {
    /// `DataStage::name()` (`dsc/dsc2.h:41`) — `ss_`'s name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.ss.name
    }
}

/// ONE `constantInfo_` ENTRY — `ConstantInfo` (`dsc/dsc2.h:44-58`), the four fields the import
/// reads. `allocations_` is dump-only and deliberately absent (see the module doc).
#[derive(Debug, Clone, PartialEq)]
pub struct ConstantInfo {
    /// `dataFormat_`.
    pub data_format: DataType,
    /// `name_`.
    pub name: String,
    /// `data_` — the fold-indexed int arrays the format encodes.
    pub data: Option<FoldedIntArray>,
    /// `isDataSymbolic_`.
    pub is_data_symbolic: bool,
}

/// ONE `computeOp_` ENTRY — `ComputeOpInfo` (`dsc/dscdefn.h:492-512`) as the import reads it
/// (`dsc/designSpaceConfig.cpp:7308-7370`). The four labeledDs lists stay the wire's compound
/// `"name-idxN"` strings: resolving them against `labeledDs_` is the adapter's lookup, done with
/// the whole op in hand the way the C++'s second pass does (`:7461-7502`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeOp {
    /// `exUnit`.
    pub ex_unit: SenComponent,
    /// `opFuncName`.
    pub op_func_name: WireOpFunc,
    /// `attributes_.dataFormat_`.
    pub data_format: DataType,
    /// `attributes_.fidelity_`.
    pub fidelity: Fidelity,
    /// `location`.
    pub location: WireLoopName,
    /// `auxLoopName`.
    pub aux_loop_name: String,
    /// `isAtMainLoop`.
    pub is_at_main_loop: bool,
    /// `isAtTop`.
    pub is_at_top: bool,
    /// `level`.
    pub level: i64,
    /// `coreExclude`.
    pub core_exclude: Vec<i64>,
    /// `coreClExclude` — `<core, cl>` pairs.
    pub core_cl_exclude: Vec<(i64, i64)>,
    /// `opConsts` — name → its four values.
    pub op_consts: BTreeMap<String, Vec<i64>>,
    /// `inputLabeledDs`.
    pub input_labeled_ds: Vec<String>,
    /// `interimLabeledDs`.
    pub interim_labeled_ds: Vec<String>,
    /// `outputLabeledDs`.
    pub output_labeled_ds: Vec<String>,
    /// `indirectAccessIndexLabeledDs`.
    pub indirect_access_index_labeled_ds: Vec<String>,
}

// -- serde shapes --------------------------------------------------------------

use super::PrimaryDim;

/// READ ONE `DataStructDims` — `DataStructDims::importJsonObj` (`dsc/dims.cpp:311-435`).
pub(super) fn data_struct_dims(json: &serde_json::Value) -> Result<DataStructDims, Refusal> {
    let mut dims = DataStructDims {
        name: String::new(),
        in_: -1.0,
        out_: -1.0,
        mb_: -1.0,
        ij_: -1.0,
        rc_: -1.0,
        kij_: -1.0,
        y_: -1.0,
        x_: -1.0,
        x1_: -1.0,
        sij_: -1.0,
        zij_: -1.0,
        i_: -1.0,
        j_: -1.0,
        r_: -1.0,
        c_: -1.0,
        ki_: -1.0,
        kj_: -1.0,
        si_: -1.0,
        sj_: -1.0,
        zi_: -1.0,
        zj_: -1.0,
        symbolic_dim_info: BTreeMap::new(),
        max_symbolic_volume: BTreeMap::new(),
        corelet_split: BTreeMap::new(),
        row_split: BTreeMap::new(),
        pe_sfp_split: BTreeMap::new(),
        padding_sizes: BTreeMap::new(),
    };
    let entries = json.as_object().ok_or(Refusal::MalformedOp {
        message: "dataStageParam_ entry is not an object".into(),
    })?;
    for (key, value) in entries {
        if set_plain(&mut dims, key, value) {
            // A plain double member — read.
        } else if key == "name_" {
            dims.name = value.as_str().unwrap_or_default().to_owned();
        } else if key == "symbolicDimInfo_" {
            for (dim_str, info) in value.as_object().into_iter().flatten() {
                let dim = PrimaryDim::from_spelling(dim_str).ok_or(Refusal::UnknownDim {
                    spelling: dim_str.clone(),
                    field: "symbolicDimInfo_",
                })?;
                let mut symbolic = SymbolicDimInfo::default();
                for (info_key, info_value) in info.as_object().into_iter().flatten() {
                    match info_key.as_str() {
                        "maxSize_" => symbolic.max_size = json_i64(info_value, "maxSize_")?,
                        "granularity_" => {
                            symbolic.granularity = json_i64(info_value, "granularity_")?;
                        }
                        _ => {}
                    }
                }
                dims.symbolic_dim_info.insert(dim, symbolic);
            }
        } else if key == "maxSymbolicVolume_" {
            // `ImportUtil::importData` over a `map<set<PrimaryDimTypes>, int>` — the set arrives
            // as its stringified key, which the adapter parses when it needs the dims.
            for (set_key, vol) in value.as_object().into_iter().flatten() {
                dims.max_symbolic_volume
                    .insert(set_key.clone(), json_i64(vol, "maxSymbolicVolume_")?);
            }
        } else if key == "coreletSplit_" {
            for (dim_str, split) in value.as_object().into_iter().flatten() {
                let dim = PrimaryDim::from_spelling(dim_str).ok_or(Refusal::UnknownDim {
                    spelling: dim_str.clone(),
                    field: "coreletSplit_",
                })?;
                let mut per_dim = Vec::new();
                for entry in split.as_array().into_iter().flatten() {
                    per_dim.push(wire_num(entry));
                }
                dims.corelet_split.insert(dim, per_dim);
            }
        } else if key == "peSfpSplit_" {
            for (dim_str, per_corelet) in value.as_object().into_iter().flatten() {
                let dim = PrimaryDim::from_spelling(dim_str).ok_or(Refusal::UnknownDim {
                    spelling: dim_str.clone(),
                    field: "peSfpSplit_",
                })?;
                let mut per_dim = BTreeMap::new();
                for (cl_str, splits) in per_corelet.as_object().into_iter().flatten() {
                    let cl: i64 = cl_str.parse().map_err(|_| Refusal::MalformedInteger {
                        text: cl_str.clone(),
                        field: "peSfpSplit_ corelet".into(),
                    })?;
                    let mut per_cl = BTreeMap::new();
                    for (comp_str, split) in splits.as_object().into_iter().flatten() {
                        let comp =
                            SenComponent::from_spelling(comp_str).ok_or(Refusal::UnknownSpelling {
                                spelling: comp_str.clone(),
                                field: "peSfpSplit_",
                            })?;
                        per_cl.insert(comp, wire_num(split));
                    }
                    per_dim.insert(cl, per_cl);
                }
                dims.pe_sfp_split.insert(dim, per_dim);
            }
        } else if key == "rowSplit_" {
            for (dim_str, per_corelet) in value.as_object().into_iter().flatten() {
                let dim = PrimaryDim::from_spelling(dim_str).ok_or(Refusal::UnknownDim {
                    spelling: dim_str.clone(),
                    field: "rowSplit_",
                })?;
                let mut per_dim = BTreeMap::new();
                for (cl_str, rows) in per_corelet.as_object().into_iter().flatten() {
                    let cl: i64 = cl_str.parse().map_err(|_| Refusal::MalformedInteger {
                        text: cl_str.clone(),
                        field: "rowSplit_ corelet".into(),
                    })?;
                    let mut per_cl = Vec::new();
                    for row in rows.as_array().into_iter().flatten() {
                        per_cl.push(wire_num(row));
                    }
                    per_dim.insert(cl, per_cl);
                }
                dims.row_split.insert(dim, per_dim);
            }
        } else if key == "paddingSizes_" {
            for (dim_str, pad) in value.as_object().into_iter().flatten() {
                let dim = PrimaryDim::from_spelling(dim_str).ok_or(Refusal::UnknownDim {
                    spelling: dim_str.clone(),
                    field: "paddingSizes_",
                })?;
                let mut info = DimPaddingSizes::default();
                for (pad_key, pad_value) in pad.as_object().into_iter().flatten() {
                    // The else-less chain (`dims.cpp:408-427`): unknown keys here — including
                    // the dumper's own `totalSize_` (`:294`) — are silently ignored.
                    match pad_key.as_str() {
                        "padFront_" => info.pad_front = json_i64(pad_value, "padFront_")?,
                        "padBack_" => info.pad_back = json_i64(pad_value, "padBack_")?,
                        "unneededPad_" => {
                            info.unneeded_pad = json_i64(pad_value, "unneededPad_")?;
                        }
                        "unneededPadFront_" => {
                            info.unneeded_pad_front =
                                json_i64(pad_value, "unneededPadFront_")?;
                        }
                        "unneededPadBack_" => {
                            info.unneeded_pad_back = json_i64(pad_value, "unneededPadBack_")?;
                        }
                        "stride_" => info.stride = json_i64(pad_value, "stride_")?,
                        "dilation_" => info.dilation = json_i64(pad_value, "dilation_")?,
                        "windowDim_" => {
                            let window = pad_value.as_str().unwrap_or_default();
                            // `PrimaryDimTypesCount` prints as "undefined", which is the
                            // no-window sentinel rather than a dim.
                            info.window_dim = if window == "undefined" {
                                None
                            } else {
                                Some(PrimaryDim::from_spelling(window).ok_or(
                                    Refusal::UnknownDim {
                                        spelling: window.to_owned(),
                                        field: "windowDim_",
                                    },
                                )?)
                            };
                        }
                        _ => {}
                    }
                }
                dims.padding_sizes.insert(dim, info);
            }
        } else {
            // The `DT_CHECK(0)` (`dims.cpp:429-431`) — an unknown top-level key is a refusal,
            // not a default.
            return Err(Refusal::UnknownDimsField {
                field: key.clone(),
                stage: dims.name.clone(),
            });
        }
    }
    Ok(dims)
}

    /// A `double` member off the wire — `number_value()`, which for json11 accepts a JSON number or
/// a string that parses; a shape it cannot read is 0, the way `number_value()` returns it.
fn wire_num(value: &serde_json::Value) -> f64 {
    match value {
        serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
        serde_json::Value::String(s) => s.parse().unwrap_or(0.0),
        _ => 0.0,
    }
}

/// ASSIGN ONE PLAIN DOUBLE MEMBER by its wire key — the first chain of `importJsonObj`'s
/// dispatch (`dsc/dims.cpp:316-357`). Returns `false` for a key that is not a plain member, so
/// the caller falls through to the rest of the dispatch.
fn set_plain(dims: &mut DataStructDims, field: &str, value: &serde_json::Value) -> bool {
    let member = match field {
        "in_" => &mut dims.in_,
        "out_" => &mut dims.out_,
        "mb_" => &mut dims.mb_,
        "ij_" => &mut dims.ij_,
        "rc_" => &mut dims.rc_,
        "kij_" => &mut dims.kij_,
        "y_" => &mut dims.y_,
        "x_" => &mut dims.x_,
        "x1_" => &mut dims.x1_,
        "sij_" => &mut dims.sij_,
        "zij_" => &mut dims.zij_,
        "i_" => &mut dims.i_,
        "j_" => &mut dims.j_,
        "r_" => &mut dims.r_,
        "c_" => &mut dims.c_,
        "ki_" => &mut dims.ki_,
        "kj_" => &mut dims.kj_,
        "si_" => &mut dims.si_,
        "sj_" => &mut dims.sj_,
        "zi_" => &mut dims.zi_,
        "zj_" => &mut dims.zj_,
        _ => return false,
    };
    *member = wire_num(value);
    true
}

/// READ ONE `dataStageParam_` — the id-keyed map of `DataStage`s (`dsc/dsc2.cpp:1122-1133`).
pub(super) fn data_stage_param(
    json: &serde_json::Value,
) -> Result<BTreeMap<i64, DataStage>, Refusal> {
    let mut out = BTreeMap::new();
    for (id_str, entry) in json.as_object().into_iter().flatten() {
        let id: i64 = id_str.parse().map_err(|_| Refusal::MalformedInteger {
            text: id_str.clone(),
            field: "dataStageParam_ key".into(),
        })?;
        // The import reads only `ss_` and `el_` off each entry (`dsc2.cpp:1126-1132`).
        let ss = entry
            .get("ss_")
            .map(|v| {
                data_struct_dims(v).map_err(|r| Refusal::MalformedOp {
                    message: format!("dataStageParam_ ss_: {r}"),
                })
            })
            .transpose()?
            .unwrap_or_else(empty_dims);
        let el = entry
            .get("el_")
            .map(|v| {
                data_struct_dims(v).map_err(|r| Refusal::MalformedOp {
                    message: format!("dataStageParam_ el_: {r}"),
                })
            })
            .transpose()?
            .unwrap_or_else(empty_dims);
        out.insert(id, DataStage { ss, el });
    }
    Ok(out)
}

/// A `DataStructDims` at the C++'s cleared default (`DataStructDims::clear`, every plain member
/// −1, every map empty) — for a side the entry omitted.
fn empty_dims() -> DataStructDims {
    DataStructDims {
        name: String::new(),
        in_: -1.0,
        out_: -1.0,
        mb_: -1.0,
        ij_: -1.0,
        rc_: -1.0,
        kij_: -1.0,
        y_: -1.0,
        x_: -1.0,
        x1_: -1.0,
        sij_: -1.0,
        zij_: -1.0,
        i_: -1.0,
        j_: -1.0,
        r_: -1.0,
        c_: -1.0,
        ki_: -1.0,
        kj_: -1.0,
        si_: -1.0,
        sj_: -1.0,
        zi_: -1.0,
        zj_: -1.0,
        symbolic_dim_info: BTreeMap::new(),
        max_symbolic_volume: BTreeMap::new(),
        corelet_split: BTreeMap::new(),
        row_split: BTreeMap::new(),
        pe_sfp_split: BTreeMap::new(),
        padding_sizes: BTreeMap::new(),
    }
}

/// READ ONE `constantInfo_` — the id-keyed map of the four imported fields
/// (`dsc/dsc2.cpp:1134-1151`).
pub(super) fn constant_info(
    json: &serde_json::Value,
) -> Result<BTreeMap<i64, ConstantInfo>, Refusal> {
    let mut out = BTreeMap::new();
    for (id_str, entry) in json.as_object().into_iter().flatten() {
        let id: i64 = id_str.parse().map_err(|_| Refusal::MalformedInteger {
            text: id_str.clone(),
            field: "constantInfo_ key".into(),
        })?;
        let mut constant = ConstantInfo {
            data_format: DataType::Sen169Fp16,
            name: String::new(),
            data: None,
            is_data_symbolic: false,
        };
        for (key, value) in entry.as_object().into_iter().flatten() {
            match key.as_str() {
                "dataFormat_" => {
                    let text = value.as_str().unwrap_or_default();
                    constant.data_format =
                        enums::data_format(text).ok_or(Refusal::UnknownSpelling {
                            spelling: text.to_owned(),
                            field: "constantInfo_ dataFormat_",
                        })?;
                }
                "name_" => {
                    constant.name = value.as_str().unwrap_or_default().to_owned();
                }
                "data_" => {
                    constant.data = fold_int_array(value)?;
                }
                "isDataSymbolic_" => {
                    constant.is_data_symbolic = json_i64(value, "isDataSymbolic_")? > 0;
                }
                // `allocations_` is dump-only: the import ignores it and rebuilds the map from
                // the allocate nodes' `constIdx_` (`dsc2.cpp:1836-1837`).
                _ => {}
            }
        }
        out.insert(id, constant);
    }
    Ok(out)
}

/// READ ONE `computeOp_` — the entry list (`dsc/designSpaceConfig.cpp:7308-7370`).
pub(super) fn compute_ops(json: &serde_json::Value) -> Result<Vec<ComputeOp>, Refusal> {
    let mut out = Vec::new();
    for entry in json.as_array().into_iter().flatten() {
        out.push(read_compute_op(entry)?);
    }
    Ok(out)
}

/// ONE `computeOp_` ENTRY.
fn read_compute_op(entry: &serde_json::Value) -> Result<ComputeOp, Refusal> {
    let ex_unit_str = entry.get("exUnit").and_then(|v| v.as_str()).unwrap_or("");
    let op_func_str = entry.get("opFuncName").and_then(|v| v.as_str()).unwrap_or("");
    let attributes = entry.get("attributes_");
    let data_format_str = attributes
        .and_then(|a| a.get("dataFormat_"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let fidelity_str = attributes
        .and_then(|a| a.get("fidelity_"))
        .and_then(|v| v.as_str())
        .unwrap_or("regular");
    let location_str = entry.get("location").and_then(|v| v.as_str()).unwrap_or("");
    let strings = |field: &str| -> Vec<String> {
        entry
            .get(field)
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_str())
            .map(str::to_owned)
            .collect()
    };
    let mut core_exclude = Vec::new();
    for id in entry
        .get("coreExclude")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
    {
        core_exclude.push(json_i64(id, "coreExclude")?);
    }
    let mut core_cl_exclude = Vec::new();
    for pair in entry
        .get("coreClExclude")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
    {
        // The dumper writes each pair as a two-element array (`designSpaceConfig.cpp:6701-6710`).
        let arr = pair.as_array().ok_or(Refusal::MalformedOp {
            message: "coreClExclude entry is not a pair".into(),
        })?;
        if arr.len() != 2 {
            return Err(Refusal::MalformedOp {
                message: format!("coreClExclude entry has {} values, want 2", arr.len()),
            });
        }
        core_cl_exclude.push((json_i64(&arr[0], "coreClExclude")?, json_i64(&arr[1], "coreClExclude")?));
    }
    let mut op_consts = BTreeMap::new();
    for (name, values) in entry
        .get("opConsts")
        .and_then(|v| v.as_object())
        .into_iter()
        .flatten()
    {
        // Four values each (`designSpaceConfig.cpp:6714-6727`), read through `std::stoul` —
        // unsigned, so negative wire values refuse here as they refuse there.
        let arr = values.as_array().ok_or(Refusal::MalformedOp {
            message: format!("opConsts {name:?} is not an array"),
        })?;
        let mut vals = Vec::with_capacity(arr.len());
        for value in arr {
            let int = json_i64(value, "opConsts")?;
            let unsigned = u32::try_from(int).map_err(|_| Refusal::MalformedInteger {
                text: int.to_string(),
                field: "opConsts".into(),
            })?;
            vals.push(unsigned as i64);
        }
        op_consts.insert(name.clone(), vals);
    }
    Ok(ComputeOp {
        ex_unit: SenComponent::from_spelling(ex_unit_str).ok_or(Refusal::UnknownSpelling {
            spelling: ex_unit_str.to_owned(),
            field: "exUnit",
        })?,
        op_func_name: WireOpFunc::from_spelling(op_func_str).ok_or(Refusal::UnknownSpelling {
            spelling: op_func_str.to_owned(),
            field: "opFuncName",
        })?,
        data_format: enums::data_format(data_format_str).ok_or(Refusal::UnknownSpelling {
            spelling: data_format_str.to_owned(),
            field: "attributes_ dataFormat_",
        })?,
        fidelity: Fidelity::from_spelling(fidelity_str).ok_or(Refusal::UnknownSpelling {
            spelling: fidelity_str.to_owned(),
            field: "attributes_ fidelity_",
        })?,
        location: WireLoopName::from_spelling(location_str).ok_or(Refusal::UnknownSpelling {
            spelling: location_str.to_owned(),
            field: "location",
        })?,
        aux_loop_name: entry
            .get("auxLoopName")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_owned(),
        is_at_main_loop: json_i64(
            entry.get("isAtMainLoop").unwrap_or(&serde_json::Value::Null),
            "isAtMainLoop",
        )
        .unwrap_or(1)
            != 0,
        is_at_top: json_i64(
            entry.get("isAtTop").unwrap_or(&serde_json::Value::Null),
            "isAtTop",
        )
        .unwrap_or(1)
            != 0,
        level: json_i64(
            entry.get("level").unwrap_or(&serde_json::Value::Null),
            "level",
        )
        .unwrap_or(0),
        core_exclude,
        core_cl_exclude,
        op_consts,
        input_labeled_ds: strings("inputLabeledDs"),
        interim_labeled_ds: strings("interimLabeledDs"),
        output_labeled_ds: strings("outputLabeledDs"),
        indirect_access_index_labeled_ds: strings("indirectAccessIndexLabeledDs"),
    })
}
