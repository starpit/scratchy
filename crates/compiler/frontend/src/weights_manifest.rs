// SPDX-License-Identifier: Apache-2.0
//! Per-architecture weights.json manifest.
//!
//! `crates/models/arch/configs/<arch>/weights.json` declares the shape of
//! every weight tensor the forward pass consumes, expressed in bound
//! names from the arch's config.json (e.g. `"hidden_size"`,
//! `"head_dim * num_attention_heads"`). The file is produced by the
//! `probe-weights` bin in the `scratchy-forward-compiler` crate by probing a
//! representative checkpoint's safetensors headers.
//!
//! The typed-Rust emit pass types every weight by this manifest, so
//! rustc checks the DSL's use of it against the declaration (the canonical example: Qwen3/Gemma3
//! per-head `q_norm` of shape `[head_dim]`, where a naïve op-signature
//! inference would resolve to `[heads * head_dim]` via the upstream
//! gemm). Unlike the old `weight_conventions.rs` table, the manifest
//! is per-arch data, not baked-in global knowledge.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::dim::{Dim, Shape, canonical_mul};

/// Parsed `weights.json`. Keys are dotted weight paths (e.g.
/// `"self_attn.q_proj"`), values are symbolic shapes (`Vec<Dim>`).
///
/// An optional `__packed_splits__` top-level key declares checkpoints
/// that ship fused tensors on disk (e.g. Phi-3's `self_attn.qkv_proj.weight`,
/// `mlp.gate_up_proj.weight`) instead of the per-slice logical tensors
/// the DSL references. Each entry maps the packed prefix (unindexed —
/// the codegen fans over `model.layers.{L}.…`) to an ordered list of
/// target paths; each target must itself appear in `entries`, and its
/// first-dim shape formula is the per-slice row count. At load time the
/// codegen-emitted prelude calls
/// [`GpuWeights::synthesize_packed_row_split_sizes`] per (layer, packed
/// prefix), splitting the mmap into virtual siblings before any
/// `Linear::load` runs.
#[derive(Debug, Default, Clone)]
pub struct WeightsManifest {
    pub entries: BTreeMap<String, Shape>,
    pub packed_splits: BTreeMap<String, Vec<String>>,
    /// Optional `__packed_concats__` top-level key — the reverse of `__packed_splits__`: a weight
    /// the DSL reads as ONE projection whose rows are several on-disk projections' rows, in the
    /// listed order (Splash's fused QKV: `self_attn.qkv_proj` ← q, k, v). Each part must itself
    /// appear in `entries`. An MLX-affine one loads still packed
    /// (`LinearLayer::load_affine_quant_rows_concat`).
    pub packed_concats: BTreeMap<String, Vec<String>>,
    /// Optional `__pad_to_mult8__` top-level key — list of weights to
    /// zero-pad on a specified axis to the next multiple of 8 at CPU
    /// load time. Each entry pairs a stem (matched against the
    /// indexed/unindexed loader convention, same as `entries`) with
    /// the dim to pad: `[{ "weight": "mlp.down_proj", "dim": 0 }, ...]`.
    /// Used by Qwen2.5-VL to dodge cuBLAS bf16 GEMM's K=3420 rejection
    /// — `vision_intermediate_size` rounded to the next mult of 8 lands
    /// every affected gemm on a supported algo (zero-fill rows/cols
    /// preserve math because `silu(0)·0 = 0` on the activation side
    /// and zero-row contractions are zero on the weight side).
    pub pad_to_mult8: Vec<PadHint>,
    /// Per-entry kind override. Absent ⇒ default
    /// [`ManifestEntryKind::LinearLayer`] (the standard PyTorch
    /// `nn.Linear` convention with `<prefix>.weight` / optional
    /// `<prefix>.bias` on disk). Present ⇒ the codegen routes
    /// the matching `LinearLayer` accessor to the alternate loader.
    /// See [`ManifestEntryKind`] for the variants.
    pub kinds: BTreeMap<String, ManifestEntryKind>,
    /// Optional `__bundles__` top-level key: each weight bundle the DSL passes whole to one op
    /// (`moe_block(x, mlp[layer])`), its kind and the shape of each of its roles —
    /// `"mlp": {"kind": "routed_experts", "router": [..], "gate": [..], ..}`. The typed-Rust
    /// emit pass types each role at its shape and passes them to the kind's constructor, whose
    /// signature relates them to each other and to the op's activations.
    pub bundles: BTreeMap<String, Bundle>,
}

/// One weight bundle (`__bundles__`): its kind, and its roles' shapes in the kind's role order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bundle {
    pub kind: BundleKind,
    /// Each role's shape, in [`BundleKind::roles`]' order.
    pub roles: Vec<Shape>,
}

/// What a weight bundle holds: the roles an op reads from it, each a tensor of the checkpoint. A
/// matrix is `[in, out]` (the manifest's orientation), an expert bank `[experts, in, out]`, a conv
/// `[channels, taps]` (the on-disk singleton axis dropped, as the loader drops it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleKind {
    /// Routed SwiGLU experts: the router and each expert's gate, up and down.
    RoutedExperts,
    /// [`Self::RoutedExperts`] whose routing adds a per-expert score correction
    /// (`e_score_correction_bias`).
    CorrectedExperts,
    /// Routed experts and a shared SwiGLU expert, scaled by its own sigmoid gate.
    GatedSharedExperts,
    /// Routed experts and shared SwiGLU experts, unscaled.
    SharedExperts,
    /// [`Self::SharedExperts`] whose routing adds a per-expert score correction.
    CorrectedSharedExperts,
    /// Routed un-gated (relu²) experts with a score correction, on their own width (a latent MoE's
    /// latent, or the block's input).
    UngatedExperts,
    /// Routed SwiGLU experts with a router bias and per-expert biases.
    BiasedExperts,
    /// A router of its own: its projection, the input's scale and a per-expert output scale.
    ScaledRouter,
    /// Unrouted expert banks (a [`Self::ScaledRouter`] routes them).
    ExpertBank,
    /// A Gated-DeltaNet mixer's non-projection weights.
    GatedDeltaNet,
    /// A Mamba-2 mixer's non-projection weights.
    Mamba2,
}

/// One argument of a bundle kind's constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleArg {
    /// A role, passed as itself.
    Role(&'static str),
    /// A part: the constructor named first, over these roles.
    Part(&'static str, &'static [&'static str]),
}

impl BundleKind {
    /// Every kind.
    pub const ALL: [Self; 11] = [
        Self::RoutedExperts,
        Self::CorrectedExperts,
        Self::GatedSharedExperts,
        Self::SharedExperts,
        Self::CorrectedSharedExperts,
        Self::UngatedExperts,
        Self::BiasedExperts,
        Self::ScaledRouter,
        Self::ExpertBank,
        Self::GatedDeltaNet,
        Self::Mamba2,
    ];

    /// Its name in `weights.json`, and its constructor's in the signatures.
    pub const fn name(self) -> &'static str {
        match self {
            Self::RoutedExperts => "routed_experts",
            Self::CorrectedExperts => "corrected_experts",
            Self::GatedSharedExperts => "gated_shared_experts",
            Self::SharedExperts => "shared_experts",
            Self::CorrectedSharedExperts => "corrected_shared_experts",
            Self::UngatedExperts => "ungated_experts",
            Self::BiasedExperts => "biased_experts",
            Self::ScaledRouter => "scaled_router",
            Self::ExpertBank => "expert_bank",
            Self::GatedDeltaNet => "gated_delta_net_mixer",
            Self::Mamba2 => "mamba2_mixer",
        }
    }

    /// Its constructor's arguments: each a role, or a part its own constructor builds from roles.
    pub const fn args(self) -> &'static [BundleArg] {
        use BundleArg::{Part, Role};
        const SWIGLU: BundleArg = Part("swiglu", &["shared_gate", "shared_up", "shared_down"]);
        match self {
            Self::RoutedExperts => &[Role("router"), Role("gate"), Role("up"), Role("down")],
            Self::CorrectedExperts => &[
                Role("router"),
                Role("correction"),
                Role("gate"),
                Role("up"),
                Role("down"),
            ],
            Self::GatedSharedExperts => &[
                Role("router"),
                Role("gate"),
                Role("up"),
                Role("down"),
                SWIGLU,
                Role("shared_scale"),
            ],
            Self::SharedExperts => &[
                Role("router"),
                Role("gate"),
                Role("up"),
                Role("down"),
                SWIGLU,
            ],
            Self::CorrectedSharedExperts => &[
                Role("router"),
                Role("correction"),
                Role("gate"),
                Role("up"),
                Role("down"),
                SWIGLU,
            ],
            Self::UngatedExperts => &[Role("router"), Role("correction"), Role("up"), Role("down")],
            Self::BiasedExperts => &[
                Role("router"),
                Role("router_bias"),
                Role("gate"),
                Role("up"),
                Role("down"),
                Part("expert_biases", &["gate_bias", "up_bias", "down_bias"]),
            ],
            Self::ScaledRouter => &[Role("proj"), Role("scale"), Role("per_expert_scale")],
            Self::ExpertBank => &[Role("gate"), Role("up"), Role("down")],
            Self::GatedDeltaNet => &[Role("conv"), Role("a_log"), Role("dt_bias"), Role("norm")],
            Self::Mamba2 => &[
                Role("conv"),
                Role("conv_bias"),
                Role("a_log"),
                Role("d"),
                Role("dt_bias"),
                Role("norm"),
            ],
        }
    }

    /// Its roles, in its constructor's argument order (a part's in place).
    pub fn roles(self) -> Vec<&'static str> {
        (self.args().iter())
            .flat_map(|arg| match arg {
                BundleArg::Role(role) => std::slice::from_ref(role),
                BundleArg::Part(_, roles) => roles,
            })
            .copied()
            .collect()
    }

    /// The kind `weights.json` names `name`.
    pub fn of_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.name() == name)
    }
}

/// Per-entry storage convention. The default
/// ([`ManifestEntryKind::LinearLayer`]) is the standard PyTorch
/// `nn.Linear` layout — the macro reads `<prefix>.weight` and
/// optional `<prefix>.bias` off disk. [`ManifestEntryKind::RawLinear`]
/// names an `nn.Parameter` weight (e.g. Gemma3's
/// `multi_modal_projector.mm_input_projection_weight`) that ships
/// without the `.weight` / `.bias` suffix; the codegen reads
/// `<prefix>` verbatim and the loaded layer carries no bias.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ManifestEntryKind {
    /// PyTorch `nn.Linear` convention: `<prefix>.weight` (+ optional
    /// `<prefix>.bias`).
    #[default]
    LinearLayer,
    /// PyTorch `nn.Parameter` convention: the on-disk key is
    /// `<prefix>` verbatim with no `.weight` suffix; no bias.
    RawLinear,
}

/// One zero-pad-to-mult-8 entry from `__pad_to_mult8__`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PadHint {
    /// Weight stem (same naming convention as `entries`). For
    /// indexed weights (per-block `mlp.down_proj`) the stem is
    /// repeated under each layer index by the load-time prelude.
    pub weight: String,
    /// Axis to pad: 0 or 1. 0 = rows (pad N for an `[N, K]` weight);
    /// 1 = cols (pad K for `[N, K]`, which is what `down_proj` needs).
    pub dim: usize,
}

impl WeightsManifest {
    /// Look up a weight path (as a slice of segments from
    /// `WeightTable::path`) in the manifest, joining segments with
    /// `.` to match the file's key format.
    pub fn lookup<S: AsRef<str>>(&self, path_segments: &[S]) -> Option<&Shape> {
        let dotted: Vec<&str> = path_segments.iter().map(|s| s.as_ref()).collect();
        self.entries.get(&dotted.join("."))
    }

    /// Storage convention for a weight path. Returns the manifest's
    /// declared `kind` if present; otherwise falls back to the
    /// default ([`ManifestEntryKind::LinearLayer`]).
    pub fn kind<S: AsRef<str>>(&self, path_segments: &[S]) -> ManifestEntryKind {
        let dotted: Vec<&str> = path_segments.iter().map(|s| s.as_ref()).collect();
        self.kinds
            .get(&dotted.join("."))
            .copied()
            .unwrap_or_default()
    }

    /// Empty manifest — used when an arch directory has no
    /// `weights.json` yet (arches predating the probe workflow).
    /// The typed emit refuses any weight it would have to type.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Test helper: build a manifest from a literal list of
    /// (dotted-path, shape-formula-strings). Parses each formula
    /// via the same path as [`load_file`].
    #[cfg(any(test, feature = "test-support"))]
    pub fn from_literals(entries: &[(&str, &[&str])]) -> Self {
        let mut out = BTreeMap::new();
        for (path, dims) in entries {
            let shape: Shape = dims.iter().map(|d| parse_dim(d).unwrap()).collect();
            out.insert((*path).to_string(), shape);
        }
        Self {
            entries: out,
            packed_splits: BTreeMap::new(),
            packed_concats: BTreeMap::new(),
            pad_to_mult8: Vec::new(),
            kinds: BTreeMap::new(),
            bundles: BTreeMap::new(),
        }
    }

    /// The convention set the original `weight_conventions.rs` table
    /// encoded — the "universal HF dense-attention + SwiGLU decoder"
    /// shapes every Llama/Qwen2/Gemma2/Granite test body exercises.
    /// Exposed for test call sites; real macro runs load a per-arch
    /// `weights.json` off disk instead.
    #[cfg(any(test, feature = "test-support"))]
    pub fn llama_test_conventions() -> Self {
        Self::from_literals(&[
            ("embed_tokens", &["vocab_size", "hidden_size"]),
            ("lm_head", &["hidden_size", "vocab_size"]),
            (
                "self_attn.q_proj",
                &["hidden_size", "head_dim * num_attention_heads"],
            ),
            (
                "self_attn.k_proj",
                &["hidden_size", "head_dim * num_key_value_heads"],
            ),
            (
                "self_attn.v_proj",
                &["hidden_size", "head_dim * num_key_value_heads"],
            ),
            (
                "self_attn.o_proj",
                &["head_dim * num_attention_heads", "hidden_size"],
            ),
            ("input_layernorm", &["hidden_size"]),
            ("post_attention_layernorm", &["hidden_size"]),
            ("norm", &["hidden_size"]),
            ("mlp.gate_proj", &["hidden_size", "intermediate_size"]),
            ("mlp.up_proj", &["hidden_size", "intermediate_size"]),
            ("mlp.down_proj", &["intermediate_size", "hidden_size"]),
        ])
    }
}

/// Errors produced when loading a `weights.json`.
#[derive(Debug)]
pub enum ManifestError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    /// A shape value in the JSON wasn't an array of strings.
    BadShape {
        path: PathBuf,
        weight: String,
        reason: String,
    },
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "reading {}: {source}", path.display()),
            Self::Json { path, source } => write!(f, "parsing {}: {source}", path.display()),
            Self::BadShape {
                path,
                weight,
                reason,
            } => write!(
                f,
                "{}: bad shape for `{}`: {reason}",
                path.display(),
                weight
            ),
        }
    }
}

impl std::error::Error for ManifestError {}

/// Load the `weights.json` next to the arch's config files. If the
/// file doesn't exist, returns an empty manifest — caller decides
/// whether that's OK or an error.
pub fn load_or_empty(arch_dir: &Path) -> Result<WeightsManifest, ManifestError> {
    let path = arch_dir.join("weights.json");
    if !path.exists() {
        return Ok(WeightsManifest::empty());
    }
    load_file(&path)
}

/// Load and parse an explicit path. Strict: file must exist and
/// parse cleanly.
pub fn load_file(path: &Path) -> Result<WeightsManifest, ManifestError> {
    let bytes = fs::read(path).map_err(|source| ManifestError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let raw: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|source| ManifestError::Json {
            path: path.to_path_buf(),
            source,
        })?;
    let obj = raw.as_object().ok_or_else(|| ManifestError::BadShape {
        path: path.to_path_buf(),
        weight: "<top-level>".into(),
        reason: "expected a JSON object".into(),
    })?;
    let mut entries: BTreeMap<String, Shape> = BTreeMap::new();
    let mut packed_splits: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut packed_concats: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut pad_to_mult8: Vec<PadHint> = Vec::new();
    let mut kinds: BTreeMap<String, ManifestEntryKind> = BTreeMap::new();
    let mut bundles: BTreeMap<String, Bundle> = BTreeMap::new();
    for (k, v) in obj {
        if k == "__bundles__" {
            let map = v.as_object().ok_or_else(|| ManifestError::BadShape {
                path: path.to_path_buf(),
                weight: k.clone(),
                reason: "`__bundles__` must be an object of bundles".into(),
            })?;
            for (bundle, decl) in map {
                let parsed = parse_bundle(decl).map_err(|reason| ManifestError::BadShape {
                    path: path.to_path_buf(),
                    weight: bundle.clone(),
                    reason,
                })?;
                bundles.insert(bundle.clone(), parsed);
            }
            continue;
        }
        if k == "__pad_to_mult8__" {
            let arr = v.as_array().ok_or_else(|| ManifestError::BadShape {
                path: path.to_path_buf(),
                weight: k.clone(),
                reason: "`__pad_to_mult8__` must be an array of `{weight, dim}` objects".into(),
            })?;
            for (i, item) in arr.iter().enumerate() {
                let obj = item.as_object().ok_or_else(|| ManifestError::BadShape {
                    path: path.to_path_buf(),
                    weight: k.clone(),
                    reason: format!("entry {i} is not an object"),
                })?;
                let weight = obj
                    .get("weight")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ManifestError::BadShape {
                        path: path.to_path_buf(),
                        weight: k.clone(),
                        reason: format!("entry {i} missing `weight: \"<stem>\"`"),
                    })?
                    .to_string();
                let dim = obj.get("dim").and_then(|v| v.as_u64()).ok_or_else(|| {
                    ManifestError::BadShape {
                        path: path.to_path_buf(),
                        weight: k.clone(),
                        reason: format!("entry {i} (`{weight}`) missing integer `dim` field"),
                    }
                })? as usize;
                if dim > 1 {
                    return Err(ManifestError::BadShape {
                        path: path.to_path_buf(),
                        weight: k.clone(),
                        reason: format!("entry {i} (`{weight}`): dim must be 0 or 1, got {dim}"),
                    });
                }
                pad_to_mult8.push(PadHint { weight, dim });
            }
            continue;
        }
        if k == "__packed_splits__" || k == "__packed_concats__" {
            let map = v.as_object().ok_or_else(|| ManifestError::BadShape {
                path: path.to_path_buf(),
                weight: k.clone(),
                reason: format!("`{k}` must be an object"),
            })?;
            let into = match k.as_str() {
                "__packed_splits__" => &mut packed_splits,
                _ => &mut packed_concats,
            };
            for (packed, targets_v) in map {
                let arr = targets_v
                    .as_array()
                    .ok_or_else(|| ManifestError::BadShape {
                        path: path.to_path_buf(),
                        weight: packed.clone(),
                        reason: "packed-split value must be an array of target paths".into(),
                    })?;
                let mut targets = Vec::with_capacity(arr.len());
                for (i, item) in arr.iter().enumerate() {
                    let s = item.as_str().ok_or_else(|| ManifestError::BadShape {
                        path: path.to_path_buf(),
                        weight: packed.clone(),
                        reason: format!("target {i} is not a string"),
                    })?;
                    targets.push(s.to_string());
                }
                into.insert(packed.clone(), targets);
            }
            continue;
        }
        let (shape, kind) = parse_entry(v).map_err(|reason| ManifestError::BadShape {
            path: path.to_path_buf(),
            weight: k.clone(),
            reason,
        })?;
        entries.insert(k.clone(), shape);
        if let Some(kind) = kind {
            kinds.insert(k.clone(), kind);
        }
    }
    for (fused, parts) in &packed_concats {
        if let Some(missing) = (parts.iter()).find(|p| !entries.contains_key(p.as_str())) {
            return Err(ManifestError::BadShape {
                path: path.to_path_buf(),
                weight: fused.clone(),
                reason: format!("packed-concat part `{missing}` has no entry of its own"),
            });
        }
    }
    Ok(WeightsManifest {
        entries,
        packed_splits,
        packed_concats,
        pad_to_mult8,
        kinds,
        bundles,
    })
}

/// A checkpoint's tensors' shapes by name, as its safetensors headers state them.
#[cfg(any(test, feature = "test-support"))]
pub type Header = BTreeMap<String, Vec<u64>>;

#[cfg(any(test, feature = "test-support"))]
impl Bundle {
    /// Check the declaration against a checkpoint: each role read from the tensors under `prefix`
    /// (one layer's bundle, e.g. `model.layers.0.mlp`) by [`role_in_checkpoint`], against its
    /// declared shape evaluated by `eval` (a config's bounds). Every role that differs, or that
    /// the checkpoint does not hold, is named.
    pub fn check(
        &self,
        header: &Header,
        prefix: &str,
        eval: impl Fn(&Dim) -> Option<u64>,
    ) -> Result<(), Vec<String>> {
        let mut wrong = Vec::new();
        for (role, declared) in self.kind.roles().into_iter().zip(&self.roles) {
            let want: Option<Vec<u64>> = declared.iter().map(&eval).collect();
            match (want, role_in_checkpoint(role, header, prefix)) {
                (Some(want), Ok(held)) if want == held => {}
                (Some(want), Ok(held)) => wrong.push(format!(
                    "`{role}`: declared {declared:?} = {want:?}, the checkpoint holds {held:?}"
                )),
                (None, _) => wrong.push(format!(
                    "`{role}`: {declared:?} names a bound the config lacks"
                )),
                (_, Err(why)) => wrong.push(format!("`{role}`: {why}")),
            }
        }
        match wrong.is_empty() {
            true => Ok(()),
            false => Err(wrong),
        }
    }
}

/// A bundle role's shape as a checkpoint holds it under `prefix`, in the manifest's orientation:
/// the layouts the loader reads it from. A matrix is stored `[out, in]` and read `[in, out]`; an
/// expert bank is stacked `[experts, in, out]` from per-expert `experts.{e}.<proj>.weight`
/// (`gate_proj` / `up_proj` / `down_proj`, Mixtral's `w1` / `w3` / `w2`), from a pre-stacked
/// `experts.<proj>` `[E, out, in]`, from a fused `experts.gate_up_proj` `[E, 2·I, in]`, or from
/// gpt-oss's MXFP4 `experts.<proj>_blocks` `[E, out, in / 32, 16]`; a conv `[C, 1, K]` is read
/// `[C, K]`, its singleton axis dropped.
#[cfg(any(test, feature = "test-support"))]
pub fn role_in_checkpoint(role: &str, header: &Header, prefix: &str) -> Result<Vec<u64>, String> {
    let at = |leaf: &str| header.get(&format!("{prefix}.{leaf}")).cloned();
    let first = |leaves: &[&str]| -> Result<Vec<u64>, String> {
        (leaves.iter()).find_map(|l| at(l)).ok_or_else(|| {
            let names: Vec<String> = leaves.iter().map(|l| format!("{prefix}.{l}")).collect();
            format!("none of {} in the checkpoint", names.join(", "))
        })
    };
    let transposed = |s: Vec<u64>| match s.as_slice() {
        [out, inp] => Ok(vec![*inp, *out]),
        _ => Err(format!("a matrix of rank {}", s.len())),
    };
    match role {
        "router" => transposed(first(&["gate.weight", "router.weight"])?),
        "correction" => first(&["gate.e_score_correction_bias"]),
        "router_bias" => first(&["router.bias"]),
        "proj" => transposed(first(&["proj.weight"])?),
        "scale" => first(&["scale"]),
        "per_expert_scale" => first(&["per_expert_scale"]),
        "shared_gate" | "shared_up" | "shared_down" => {
            let proj = &role["shared_".len()..];
            transposed(first(&[
                &format!("shared_expert.{proj}_proj.weight"),
                &format!("shared_experts.{proj}_proj.weight"),
            ])?)
        }
        "shared_scale" => transposed(first(&["shared_expert_gate.weight"])?),
        "gate_bias" | "up_bias" => {
            let fused = first(&["experts.gate_up_proj_bias"])?;
            match fused.as_slice() {
                [e, two_i] if two_i % 2 == 0 => Ok(vec![*e, two_i / 2]),
                _ => Err(format!("a fused gate/up bias of shape {fused:?}")),
            }
        }
        "down_bias" => first(&["experts.down_proj_bias"]),
        "gate" | "up" | "down" => expert_bank(role, header, prefix),
        "conv" => match first(&["conv1d.weight"])?.as_slice() {
            [c, 1, k] => Ok(vec![*c, *k]),
            other => Err(format!(
                "a conv of shape {other:?}, not [channels, 1, taps]"
            )),
        },
        "conv_bias" => first(&["conv1d.bias"]),
        "a_log" => first(&["A_log"]),
        "d" => first(&["D"]),
        "dt_bias" => first(&["dt_bias"]),
        "norm" => first(&["norm.weight"]),
        other => Err(format!("no checkpoint layout for the role `{other}`")),
    }
}

/// An expert bank's stacked `[experts, in, out]` ([`role_in_checkpoint`]'s layouts).
#[cfg(any(test, feature = "test-support"))]
fn expert_bank(role: &str, header: &Header, prefix: &str) -> Result<Vec<u64>, String> {
    let names: &[&str] = match role {
        "gate" => &["gate_proj", "w1"],
        "up" => &["up_proj", "w3"],
        _ => &["down_proj", "w2"],
    };
    // Per-expert: every expert's `[out, in]`, one shape.
    for name in names {
        let experts: Vec<&Vec<u64>> = (0..)
            .map_while(|e| header.get(&format!("{prefix}.experts.{e}.{name}.weight")))
            .collect();
        if let Some(first) = experts.first() {
            if let Some(other) = experts.iter().find(|s| *s != first) {
                return Err(format!("experts of two shapes, {first:?} and {other:?}"));
            }
            return match first.as_slice() {
                [out, inp] => Ok(vec![experts.len() as u64, *inp, *out]),
                _ => Err(format!("an expert matrix of shape {first:?}")),
            };
        }
    }
    let stacked = |leaf: &str| header.get(&format!("{prefix}.experts.{leaf}"));
    // Pre-stacked, one projection.
    for name in names {
        if let Some(s) = stacked(name) {
            return match s.as_slice() {
                [e, out, inp] => Ok(vec![*e, *inp, *out]),
                _ => Err(format!("a stacked bank of shape {s:?}")),
            };
        }
    }
    // MXFP4 blocks: 32 codes a block.
    let blocks = |leaf: &str| match stacked(leaf).map(Vec::as_slice) {
        Some([e, out, nb, 16]) => Some((*e, *out, nb * 32)),
        _ => None,
    };
    let (fused, fused_blocks) = (stacked("gate_up_proj"), blocks("gate_up_proj_blocks"));
    match role {
        "gate" | "up" => {
            if let Some([e, two_i, inp]) = fused.map(Vec::as_slice) {
                return Ok(vec![*e, *inp, two_i / 2]);
            }
            if let Some((e, two_i, inp)) = fused_blocks {
                return Ok(vec![e, inp, two_i / 2]);
            }
        }
        _ => {
            if let Some((e, out, inp)) = blocks("down_proj_blocks") {
                return Ok(vec![e, inp, out]);
            }
        }
    }
    Err(format!("no `{role}` expert bank under {prefix}.experts"))
}

/// One `__bundles__` declaration: `{"kind": "<kind>", "<role>": [dims], ..}`, every role of the
/// kind and no other.
fn parse_bundle(v: &serde_json::Value) -> Result<Bundle, String> {
    let obj = (v.as_object()).ok_or("a bundle is an object of its kind and roles")?;
    let name = (obj.get("kind").and_then(|k| k.as_str())).ok_or("a bundle names its `kind`")?;
    let kind = BundleKind::of_name(name).ok_or_else(|| {
        let known: Vec<&str> = BundleKind::ALL.iter().map(|k| k.name()).collect();
        format!("unknown bundle kind `{name}` (known: {})", known.join(", "))
    })?;
    if let Some(extra) = (obj.keys()).find(|r| *r != "kind" && !kind.roles().contains(&r.as_str()))
    {
        return Err(format!(
            "`{name}` has no role `{extra}` (its roles: {:?})",
            kind.roles()
        ));
    }
    let roles = (kind.roles().iter())
        .map(|role| {
            let shape = (obj.get(*role)).ok_or_else(|| format!("`{name}` needs its `{role}`"))?;
            parse_shape(shape).map_err(|e| format!("`{role}`: {e}"))
        })
        .collect::<Result<_, String>>()?;
    Ok(Bundle { kind, roles })
}

/// Parse one weight entry value. Two accepted shapes:
///   - Array of dim strings (`["hidden_size", "num_heads * head_dim"]`)
///     — the legacy default; entry kind defaults to `LinearLayer`.
///   - Object `{ "shape": [...], "kind": "raw_linear" }` — opt-in
///     storage convention override. Today the only non-default
///     kind is `raw_linear` (Gemma3 MM projector's
///     `mm_input_projection_weight` is an `nn.Parameter`, not a
///     `nn.Linear`).
fn parse_entry(v: &serde_json::Value) -> Result<(Shape, Option<ManifestEntryKind>), String> {
    if v.is_array() {
        return Ok((parse_shape(v)?, None));
    }
    let obj = v.as_object().ok_or_else(|| {
        "expected an array of shape strings or `{shape, kind}` object".to_string()
    })?;
    let shape_v = obj
        .get("shape")
        .ok_or_else(|| "object form requires a `shape` field".to_string())?;
    let shape = parse_shape(shape_v)?;
    let kind = match obj.get("kind").and_then(|k| k.as_str()) {
        Some("raw_linear") => Some(ManifestEntryKind::RawLinear),
        Some(other) => return Err(format!("unknown manifest entry kind `{other}`")),
        None => None,
    };
    Ok((shape, kind))
}

/// Parse a JSON shape value — must be an array of strings, each
/// string either a bound name (`"hidden_size"`) or a `*`-separated
/// product (`"head_dim * num_attention_heads"`).
fn parse_shape(v: &serde_json::Value) -> Result<Shape, String> {
    let arr = v
        .as_array()
        .ok_or_else(|| "expected an array of shape strings".to_string())?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        let s = item
            .as_str()
            .ok_or_else(|| format!("dim {i} is not a string"))?;
        out.push(parse_dim(s)?);
    }
    Ok(out)
}

fn parse_dim(s: &str) -> Result<Dim, String> {
    let factors: Vec<Dim> = s
        .split('*')
        .map(str::trim)
        .map(|f| {
            if f.is_empty() {
                return Err("empty factor in product".to_string());
            }
            if let Ok(n) = f.parse::<u64>() {
                Ok(Dim::Lit(n))
            } else {
                Ok(Dim::Bound(f.to_string()))
            }
        })
        .collect::<Result<_, _>>()?;
    match factors.len() {
        0 => Err("empty shape string".to_string()),
        1 => Ok(factors.into_iter().next().unwrap()),
        _ => Ok(canonical_mul(factors)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dim::Dim;

    #[test]
    fn parses_single_bound() {
        assert_eq!(
            parse_dim("hidden_size").unwrap(),
            Dim::Bound("hidden_size".into())
        );
    }

    #[test]
    fn parses_product_of_two_bounds_canonically() {
        // Product form is canonicalised — bound names sorted
        // alphabetically so `head_dim * num_attention_heads` produces
        // the same `Dim` as `num_attention_heads * head_dim`.
        let a = parse_dim("head_dim * num_attention_heads").unwrap();
        let b = parse_dim("num_attention_heads * head_dim").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn parses_integer_literal_factor() {
        let d = parse_dim("2 * head_dim").unwrap();
        match d {
            Dim::Mul(fs) => {
                assert!(fs.iter().any(|f| matches!(f, Dim::Lit(2))));
                assert!(
                    fs.iter()
                        .any(|f| matches!(f, Dim::Bound(b) if b == "head_dim"))
                );
            }
            _ => panic!("expected Mul"),
        }
    }

    #[test]
    fn lookup_joins_segments_with_dots() {
        let mut m = WeightsManifest::empty();
        m.entries.insert(
            "self_attn.q_proj".into(),
            vec![Dim::Bound("hidden_size".into())],
        );
        let got = m.lookup(&["self_attn", "q_proj"]);
        assert!(got.is_some());
    }

    #[test]
    fn missing_file_returns_empty_manifest() {
        let tmp = std::env::temp_dir().join("scratchy_weights_manifest_test_nonexistent");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let m = load_or_empty(&tmp).unwrap();
        assert!(m.entries.is_empty());
        assert!(m.pad_to_mult8.is_empty());
    }

    #[test]
    fn parses_pad_to_mult8_top_level_array() {
        // G.6.5 contract: `__pad_to_mult8__` is a top-level array of
        // `{ "weight": "...", "dim": 0|1 }` objects. Order is
        // preserved, dim is parsed as usize, and other top-level keys
        // (regular weight stems) coexist cleanly.
        let tmp = std::env::temp_dir().join("scratchy_pad_mult8_manifest_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join("weights.json");
        std::fs::write(
            &path,
            r#"{
              "__pad_to_mult8__": [
                {"weight": "mlp.gate_proj", "dim": 0},
                {"weight": "mlp.up_proj", "dim": 0},
                {"weight": "mlp.down_proj", "dim": 1}
              ],
              "norm1": ["vision_embed_dim"]
            }"#,
        )
        .unwrap();
        let m = load_file(&path).expect("load_file");
        assert_eq!(m.pad_to_mult8.len(), 3);
        assert_eq!(
            m.pad_to_mult8[0],
            PadHint {
                weight: "mlp.gate_proj".into(),
                dim: 0
            }
        );
        assert_eq!(
            m.pad_to_mult8[2],
            PadHint {
                weight: "mlp.down_proj".into(),
                dim: 1
            }
        );
        // Coexists with regular entries.
        assert!(m.entries.contains_key("norm1"));
    }

    #[test]
    fn parses_raw_linear_object_form() {
        // G.7(a) contract: a manifest value of the form
        // `{ "shape": [...], "kind": "raw_linear" }` parses cleanly,
        // populates `entries` with the shape AND `kinds` with the
        // override. Array-form entries leave `kinds` absent (default
        // is `LinearLayer`, exposed via `WeightsManifest::kind`).
        let tmp = std::env::temp_dir().join("scratchy_raw_linear_manifest_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join("weights.json");
        std::fs::write(
            &path,
            r#"{
              "self_attn.q_proj": ["hidden_size", "hidden_size"],
              "mm.mm_input_projection_weight": {
                "shape": ["vision_embed_dim", "d_model"],
                "kind": "raw_linear"
              }
            }"#,
        )
        .unwrap();
        let m = load_file(&path).expect("load_file");
        // Both entries land in `entries`.
        assert!(m.entries.contains_key("self_attn.q_proj"));
        assert!(m.entries.contains_key("mm.mm_input_projection_weight"));
        // Kind override only on the object-form entry.
        assert_eq!(
            m.kind(&["mm", "mm_input_projection_weight"]),
            ManifestEntryKind::RawLinear
        );
        assert_eq!(
            m.kind(&["self_attn", "q_proj"]),
            ManifestEntryKind::LinearLayer
        );
        // Absent path → default kind.
        assert_eq!(
            m.kind(&["does", "not", "exist"]),
            ManifestEntryKind::LinearLayer
        );
    }

    #[test]
    fn rejects_unknown_manifest_entry_kind() {
        let tmp = std::env::temp_dir().join("scratchy_bad_kind_manifest_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join("weights.json");
        std::fs::write(
            &path,
            r#"{ "x": { "shape": ["d"], "kind": "totally_made_up" } }"#,
        )
        .unwrap();
        let err = load_file(&path).expect_err("unknown kind must reject");
        let msg = format!("{err}");
        assert!(msg.contains("unknown manifest entry kind"), "got: {msg}");
    }

    #[test]
    fn pad_to_mult8_rejects_dim_out_of_range() {
        let tmp = std::env::temp_dir().join("scratchy_pad_mult8_bad_dim_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join("weights.json");
        std::fs::write(
            &path,
            r#"{ "__pad_to_mult8__": [{"weight": "x", "dim": 7}] }"#,
        )
        .unwrap();
        let err = load_file(&path).expect_err("dim=7 is out of range");
        let msg = format!("{err}");
        assert!(msg.contains("dim must be 0 or 1"), "got: {msg}");
    }

    /// Every `<prefix>.bias` in `manifest` whose declaration is not one dim equal to the declared
    /// `<prefix>`'s last (its output width), or whose `<prefix>` is undeclared.
    fn biases_off_their_weight(manifest: &WeightsManifest) -> Vec<String> {
        (manifest.entries.iter())
            .filter_map(|(key, bias)| {
                let prefix = key.strip_suffix(".bias")?;
                let out = (manifest.entries.get(prefix)).and_then(|w| w.last());
                let agrees = out.is_some_and(|out| bias.as_slice() == std::slice::from_ref(out));
                (!agrees).then(|| format!("{key}: {bias:?} vs `{prefix}`'s {out:?}"))
            })
            .collect()
    }

    /// A bias states its weight's output width a second time, so the two must agree in every
    /// checked-in `weights.json`, typed carrier or not: fixing a weight's declaration without its
    /// bias's is a failure here.
    #[test]
    fn every_declared_bias_is_its_weights_output_width() {
        let configs =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/arch/configs");
        let mut checked = 0;
        let mut drift = Vec::new();
        for dir in std::fs::read_dir(&configs).expect("configs") {
            let path = dir.expect("entry").path().join("weights.json");
            if !path.exists() {
                continue;
            }
            let manifest = load_file(&path).expect("manifest");
            checked += (manifest.entries.keys())
                .filter(|k| k.ends_with(".bias"))
                .count();
            drift.extend(
                (biases_off_their_weight(&manifest).into_iter())
                    .map(|d| format!("{}: {d}", path.display())),
            );
        }
        assert!(
            checked > 0,
            "no bias declarations under {}",
            configs.display()
        );
        assert!(drift.is_empty(), "{drift:#?}");
    }

    /// The check fires: a bias left at its weight's old width, and a bias with no weight.
    #[test]
    fn a_bias_off_its_weights_output_width_is_reported() {
        let agree = WeightsManifest::from_literals(&[
            ("q_proj", &["hidden_size", "head_dim * num_attention_heads"]),
            ("q_proj.bias", &["head_dim * num_attention_heads"]),
        ]);
        assert!(biases_off_their_weight(&agree).is_empty());
        let drifted = WeightsManifest::from_literals(&[
            ("q_proj", &["hidden_size", "head_dim * num_attention_heads"]),
            ("q_proj.bias", &["hidden_size"]),
            ("orphan.bias", &["hidden_size"]),
        ]);
        let found = biases_off_their_weight(&drifted);
        assert_eq!(found.len(), 2, "{found:#?}");
        assert!(
            found[0].starts_with("orphan.bias") && found[1].starts_with("q_proj.bias"),
            "{found:#?}"
        );
    }

    /// A bundle declaration parses into its kind's roles, in the kind's order whatever the JSON's.
    #[test]
    fn a_bundle_parses_its_roles_in_its_kinds_order() {
        let decl = serde_json::json!({
            "down": ["num_experts", "moe_intermediate_size", "hidden_size"],
            "kind": "routed_experts",
            "router": ["hidden_size", "num_experts"],
            "up": ["num_experts", "hidden_size", "moe_intermediate_size"],
            "gate": ["num_experts", "hidden_size", "moe_intermediate_size"],
        });
        let bundle = parse_bundle(&decl).unwrap();
        assert_eq!(bundle.kind, BundleKind::RoutedExperts);
        let ranks: Vec<usize> = bundle.roles.iter().map(Vec::len).collect();
        assert_eq!(ranks, [2, 3, 3, 3], "router, gate, up, down");
        assert_eq!(bundle.roles[3][2], Dim::Bound("hidden_size".into()));
    }

    /// A bundle of an unknown kind, missing one of its kind's roles, or naming a role its kind
    /// lacks is refused, each naming what is wrong.
    #[test]
    fn a_bundle_without_exactly_its_kinds_roles_is_refused() {
        let routed = |roles: serde_json::Value| {
            let mut decl = roles;
            decl["kind"] = "routed_experts".into();
            parse_bundle(&decl).unwrap_err()
        };
        let shape = serde_json::json!(["hidden_size"]);
        let unknown = parse_bundle(&serde_json::json!({"kind": "experts"})).unwrap_err();
        assert!(
            unknown.contains("unknown bundle kind `experts`"),
            "{unknown}"
        );
        for kind in BundleKind::ALL {
            assert!(
                unknown.contains(kind.name()),
                "names `{}`: {unknown}",
                kind.name()
            );
        }
        let missing = routed(serde_json::json!({"router": shape, "gate": shape, "up": shape}));
        assert!(missing.contains("needs its `down`"), "{missing}");
        let extra = routed(serde_json::json!({
            "router": shape, "gate": shape, "up": shape, "down": shape, "bias": shape,
        }));
        assert!(extra.contains("has no role `bias`"), "{extra}");
    }

    /// Every kind's name round-trips, and no two kinds share one.
    #[test]
    fn every_bundle_kind_is_named_once() {
        for kind in BundleKind::ALL {
            assert_eq!(BundleKind::of_name(kind.name()), Some(kind));
        }
        let names: std::collections::BTreeSet<&str> =
            BundleKind::ALL.iter().map(|k| k.name()).collect();
        assert_eq!(names.len(), BundleKind::ALL.len());
    }
}
