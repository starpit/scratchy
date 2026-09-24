//! THE WIRE'S FOLD DATA — a `FoldManager` as dbo's `dumpProgram` prints it.
//!
//! `print` (`util/foldManager/foldInfrastructure.h:134-160` roughly) writes
//! `{dim_prop_func: [...], dim_prop_attr: [...], data_: {...}}`, and `importFromJson`
//! (`util/foldManager/foldInfrastructure.h:2767-2825`) reads exactly those three keys:
//!
//! - `dim_prop_attr` — the `FoldDimProp`s: one `{factor_, label_}` per folded dimension.
//! - `dim_prop_func` — how each dimension's value varies with its index: `Const`, `Map` or
//!   `Affine {alpha_, beta_}`. The importer refuses anything else
//!   (`Func type not yet supported in import function`), so the reader does too.
//! - `data_` — the per-coordinate values, keyed by a JSON-stringified int deque like `"[0, 0, 0]"`.
//!
//! ⭐ ZERO-FOLD IS THE WHOLE-VALUE FORM. Where there are no fold dimensions, `importFromJson`'s
//! early arm reads the JSON *itself* as the value — `startAddr_: "0"` in a `DataInfo` is a plain
//! string, not the three-key object (`foldInfrastructure.h:2771-2774`). Both shapes are carried:
//! [`FoldData::ZeroFold`] for the value and [`FoldData::Folded`] for the three-key object.
//!
//! ⚠️ AFFINE `beta` IS IMPORTED BUT NOT COMPENSATED. The reference's own TODO
//! (`foldInfrastructure.h:2804-2807`): affine folds mixed with map/const dims keep their `beta`
//! in the data values, and a mixed affine/data import is refused outright
//! (`:2809-2812`). The reader carries `alpha`/`beta` faithfully and makes the same refusal.

use std::collections::BTreeMap;

use serde::Deserialize;

use super::Refusal;

/// ONE FOLDED DIMENSION'S SHAPE — `FoldDimProp` (`util/foldManager/foldInfrastructure.h:119-150`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoldDimProp {
    /// `factor_` — how many values the dimension takes.
    pub factor: u32,
    /// `label_` — `"core"`, `"corelet"`, `"time"`, `"core_fold"`, … free-form, because the C++ field
    /// is a free-form `std::string`.
    pub label: String,
}

/// HOW ONE DIMENSION'S VALUE VARIES WITH ITS INDEX — the `dim_prop_func` entry, a one-key object
/// keyed by `FoldInfraUtils::stringToBaseFuncType` (`foldInfrastructure.h:2786-2795`).
///
/// ⛔ THE IMPORTER ACCEPTS THREE AND REFUSES THE REST (`:2793-2795`); `WkSplit` and `Unknown` are
/// `BaseFuncType`s the wire cannot carry here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldDimFunc {
    /// `Const {}` — one value for the whole dimension.
    Const,
    /// `Map {}` — the per-index table in `data_`.
    Map,
    /// `Affine {alpha_, beta_}` — `value = alpha * index + beta`.
    Affine {
        /// `alpha_`.
        alpha: i64,
        /// `beta_`.
        beta: i64,
    },
}

impl FoldDimFunc {
    /// `FoldInfraUtils::stringToBaseFuncType` (`util/foldManager/foldInfrastructure.h`), restricted
    /// to the three the importer accepts — the parse boundary. An unknown or unsupported spelling
    /// is `None`.
    #[must_use]
    pub fn from_spelling(kind: &str) -> Option<Self> {
        Some(match kind {
            "Const" => Self::Const,
            "Map" => Self::Map,
            "Affine" => Self::Affine { alpha: 0, beta: 0 },
            _ => return None,
        })
    }
}

/// ONE FOLD-INDEXED VALUE — either the whole value (zero fold dims) or the three-key object.
///
/// The `data_` values are strings on the wire even where they are integers
/// (`ImportUtil::importData` accepts both, `util/import_utils.h:37-46`), because a start address
/// can be symbolic — `"0"` in the fixtures, but a symbol name in a program with
/// `isStartAddrSymbolic_` set. The payload is kept as the string it arrived as; interpreting it is
/// the adapter's decision, made per field with the field's `isStartAddrSymbolic_` in hand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FoldData {
    /// The zero-fold form: the JSON value itself (`foldInfrastructure.h:2771-2774`).
    ZeroFold(String),
    /// The folded form: the three-key object.
    Folded(FoldedData),
}

/// THE THREE-KEY FOLDED FORM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoldedData {
    /// `dim_prop_attr` — one per folded dimension, outermost first.
    pub props: Vec<FoldDimProp>,
    /// `dim_prop_func` — how each dimension varies, parallel to `props`.
    pub funcs: Vec<FoldDimFunc>,
    /// `data_` — the values, keyed by the coordinate deque as printed (`"[0, 0, 0]"`).
    pub data: BTreeMap<String, String>,
}

impl FoldedData {
    /// READ ONE `data_` KEY AS A COORDINATE — `importData<std::deque<int64_t>>(coordStr)`
    /// (`foldInfrastructure.h:2815-2818`), which parses the JSON-stringified deque the dumper
    /// wrote. A key that is not a JSON int array is a refusal, matching the `DT_CHECK_MSG` the
    /// coordinate-count check raises for the malformed ones.
    fn coordinate(key: &str) -> Result<Vec<i64>, Refusal> {
        let parsed: Vec<i64> = serde_json::from_str(key)
            .map_err(|_| Refusal::MalformedCoordinate { key: key.to_owned() })?;
        Ok(parsed)
    }
}

// -- serde shapes --------------------------------------------------------------
//
// The wire is json11's output, which is a strict JSON subset. The shapes below are untagged so the
// zero-fold value and the three-key object both parse into FoldData.

/// The untagged wire form of [`FoldData`].
#[derive(Deserialize)]
#[serde(untagged)]
enum WireFold {
    /// The three-key object.
    Folded {
        dim_prop_func: Vec<serde_json::Value>,
        dim_prop_attr: Vec<WireProp>,
        #[serde(rename = "data_", default)]
        data: BTreeMap<String, String>,
    },
    /// A plain string — the zero-fold form (`startAddr_` is `"0"` in the fixtures).
    Plain(String),
}

/// READ ONE `dim_prop_func` ENTRY — the C++ reads it as a ONE-KEY object whose key is the func
/// type (`foldInfrastructure.h:2789-2795`: `DT_CHECK(funcTypeStruct.size() == 1)`, then
/// `stringToBaseFuncType.at(funcTypeStr)`), so the reader does the same. `{"Const": {}}` and
/// `{"Map": {}}` are recognized by their key alone; `{"Affine": {"alpha_": .., "beta_": ..}}`
/// carries metadata.
fn fold_func(entry: &serde_json::Value) -> Result<FoldDimFunc, Refusal> {
    let map = entry.as_object().ok_or(Refusal::MalformedFold {
        message: "dim_prop_func entry is not an object".into(),
    })?;
    if map.len() != 1 {
        return Err(Refusal::MalformedFold {
            message: format!("dim_prop_func entry has {} keys, want 1", map.len()),
        });
    }
    let (kind, meta) = map.iter().next().expect("len checked above");
    match kind.as_str() {
        // `baseFuncTypeToString` (`foldInfrastructure.h:48-51`).
        "Const" => Ok(FoldDimFunc::Const),
        "Map" => Ok(FoldDimFunc::Map),
        "Affine" => {
            // `alpha_`/`beta_` arrive as a number or a string — `ImportUtil::importData` over an
            // integral type (`util/import_utils.h:37-46`).
            let value = |field: &str| -> Result<i64, Refusal> {
                let raw = meta.get(field).ok_or(Refusal::MalformedFold {
                    message: format!("Affine without {field}"),
                })?;
                raw.as_i64()
                    .or_else(|| raw.as_str().and_then(|s| s.parse().ok()))
                    .ok_or(Refusal::MalformedFold {
                        message: format!("Affine {field} is not an integer"),
                    })
            };
            Ok(FoldDimFunc::Affine {
                alpha: value("alpha_")?,
                beta: value("beta_")?,
            })
        }
        // "Func type not yet supported in import function" (`:2793-2795`) — `WkSplit` and
        // anything else the C++ would also refuse.
        other => Err(Refusal::MalformedFold {
            message: format!("fold func type {other:?} not supported in import"),
        }),
    }
}

/// One `dim_prop_attr` entry.
#[derive(Deserialize)]
struct WireProp {
    #[serde(rename = "factor_")]
    factor: Option<WireInt>,
    #[serde(rename = "label_", default)]
    label: String,
}

/// An integer that may arrive as a number or a string.
#[derive(Deserialize)]
#[serde(untagged)]
enum WireInt {
    /// As a JSON number.
    Num(i64),
    /// As a JSON string.
    Text(String),
}

impl WireInt {
    /// `ImportUtil::importData` over an integral type.
    fn value(&self, field: &'static str) -> Result<i64, Refusal> {
        match self {
            Self::Num(n) => Ok(*n),
            Self::Text(s) => s.parse().map_err(|_| Refusal::MalformedInteger {
                text: s.clone(),
                field: field.into(),
            }),
        }
    }
}

impl TryFrom<WireFold> for FoldData {
    type Error = Refusal;

    fn try_from(wire: WireFold) -> Result<Self, Refusal> {
        match wire {
            WireFold::Plain(s) => Ok(FoldData::ZeroFold(s)),
            WireFold::Folded {
                dim_prop_func,
                dim_prop_attr,
                data,
            } => {
                // "Different number of dims between json and caller" (`:2779-2780`) — here the two
                // wire arrays must agree with each other.
                if dim_prop_func.len() != dim_prop_attr.len() {
                    return Err(Refusal::FoldDimMismatch {
                        funcs: dim_prop_func.len(),
                        props: dim_prop_attr.len(),
                    });
                }
                let mut funcs = Vec::with_capacity(dim_prop_func.len());
                for entry in &dim_prop_func {
                    funcs.push(fold_func(entry)?);
                }
                let mut props = Vec::with_capacity(dim_prop_attr.len());
                for prop in &dim_prop_attr {
                    props.push(FoldDimProp {
                        factor: u32::try_from(
                            prop.factor
                                .as_ref()
                                .map(|w| w.value("factor_"))
                                .transpose()?
                                .unwrap_or(0),
                        )
                        .map_err(|_| Refusal::NegativeFoldFactor)?,
                        label: prop.label.clone(),
                    });
                }
                // "Num of dimensions in coordinate not matching num folds" (`:2817-2818`).
                for key in data.keys() {
                    let coord = FoldedData::coordinate(key)?;
                    if coord.len() != props.len() {
                        return Err(Refusal::FoldCoordMismatch {
                            key: key.clone(),
                            coord: coord.len(),
                            folds: props.len(),
                        });
                    }
                }
                Ok(FoldData::Folded(FoldedData {
                    props,
                    funcs,
                    data,
                }))
            }
        }
    }
}

/// PARSE ONE FOLD VALUE — the boundary. `None` for a JSON `null` (fields the dumper omits when
/// empty); a refusal for a shape the importer would reject.
pub(crate) fn fold_data(json: &serde_json::Value) -> Result<Option<FoldData>, Refusal> {
    match json {
        serde_json::Value::Null => Ok(None),
        other => WireFold::deserialize(other)
            .map_err(|e| Refusal::MalformedFold {
                message: e.to_string(),
            })?
            .try_into()
            .map(Some),
    }
}
