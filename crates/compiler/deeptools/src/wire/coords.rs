//! THE WIRE'S COORDINATE MAPS — `Coordinates::dsc_import_json` (`dsc/dsc2.h:324-359`).
//!
//! A node's `coordinates_` is `{coordInfo: {dim: {spatial, temporal, elemArr, padding, folds}},
//! coreIdToWkSlice_: {core: {dim: int}}}`. The fold manager under `folds` is the same three-key
//! object [`super::fold`] reads; the per-dim counts and padding are plain values.

use std::collections::BTreeMap;

use serde::Deserialize;

use super::enums::PadType;
use super::fold::{fold_data, FoldData};
use super::PrimaryDim;
use super::Refusal;

/// ONE DIMENSION'S COORDINATE INFO — the `coordInfo` entry (`dsc/dsc2.h:326-346`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoordInfo {
    /// `spatial` — `numOfSpatialFolds_[dim]`.
    pub spatial: i64,
    /// `temporal` — `numOfTemporalFolds_[dim]`.
    pub temporal: i64,
    /// `elemArr` — `numOfElemArrFolds_[dim]`.
    pub elem_arr: i64,
    /// `padding` — `padding_` for the dim.
    pub padding: PadType,
    /// `folds` — the fold manager for the dim.
    pub folds: FoldData,
}

/// A NODE'S COORDINATES — `printCoordinates`/`dsc_import_json` (`dsc/dsc2.h:258-359`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Coordinates {
    /// `coordInfo`, one entry per dimension that has one.
    pub coord_info: BTreeMap<PrimaryDim, CoordInfo>,
    /// `coreIdToWkSlice_` — which work-slice each core takes, per dim.
    pub core_id_to_wk_slice: BTreeMap<i64, BTreeMap<PrimaryDim, i64>>,
}

/// The wire form of [`Coordinates`].
#[derive(Deserialize)]
pub(super) struct WireCoordinates {
    #[serde(rename = "coordInfo")]
    coord_info: BTreeMap<String, WireCoordInfo>,
    #[serde(rename = "coreIdToWkSlice_")]
    core_id_to_wk_slice: BTreeMap<String, BTreeMap<String, i64>>,
}

/// The wire form of one `coordInfo` entry.
#[derive(Deserialize)]
struct WireCoordInfo {
    spatial: i64,
    temporal: i64,
    #[serde(rename = "elemArr")]
    elem_arr: i64,
    padding: String,
    folds: serde_json::Value,
}

impl TryFrom<WireCoordinates> for Coordinates {
    type Error = Refusal;

    fn try_from(wire: WireCoordinates) -> Result<Self, Refusal> {
        let mut coord_info = BTreeMap::new();
        for (dim_str, info) in wire.coord_info {
            let dim = PrimaryDim::from_spelling(&dim_str).ok_or(Refusal::UnknownDim {
                spelling: dim_str,
                field: "coordInfo",
            })?;
            let padding = PadType::from_spelling(&info.padding).ok_or(Refusal::UnknownSpelling {
                spelling: info.padding,
                field: "padding",
            })?;
            let folds = fold_data(&info.folds)?.unwrap_or(FoldData::ZeroFold(String::new()));
            coord_info.insert(
                dim,
                CoordInfo {
                    spatial: info.spatial,
                    temporal: info.temporal,
                    elem_arr: info.elem_arr,
                    padding,
                    folds,
                },
            );
        }
        let mut core_id_to_wk_slice = BTreeMap::new();
        for (core_str, slices) in wire.core_id_to_wk_slice {
            let core: i64 = core_str.parse().map_err(|_| Refusal::MalformedInteger {
                text: core_str,
                field: "coreIdToWkSlice_ key".into(),
            })?;
            let mut per_dim = BTreeMap::new();
            for (dim_str, value) in slices {
                let dim = PrimaryDim::from_spelling(&dim_str).ok_or(Refusal::UnknownDim {
                    spelling: dim_str,
                    field: "coreIdToWkSlice_",
                })?;
                per_dim.insert(dim, value);
            }
            core_id_to_wk_slice.insert(core, per_dim);
        }
        Ok(Self {
            coord_info,
            core_id_to_wk_slice,
        })
    }
}

/// PARSE ONE `coordinates_` FIELD — the boundary. `None` for a JSON `null`.
pub(super) fn coordinates(json: &serde_json::Value) -> Result<Option<Coordinates>, Refusal> {
    match json {
        serde_json::Value::Null => Ok(None),
        other => WireCoordinates::deserialize(other)
            .map_err(|e| Refusal::MalformedCoordinates {
                message: e.to_string(),
            })?
            .try_into()
            .map(Some),
    }
}
