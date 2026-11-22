// SPDX-License-Identifier: Apache-2.0
//! THE KERNELS ARE THE SPEC: every kernel in `backends/spyre/kernels/`, at the shapes the bake
//! compiles it at (splice's case table), lifts — or the test states, by name, why it does not yet.
//!
//! [`EXPECTED`] is the red list. A kernel that starts lifting fails the test until its entry says
//! `Lifts`, and a kernel that stops lifting fails it too, so the list is always the state of the
//! code. Every entry point a kernel file defines must appear in the table: a new kernel with no
//! entry fails the test. A kernel whose program walks must walk exactly once.

use std::collections::{BTreeSet, HashMap};

use ktir_core::arena::Arena;
use ktir_superdsc::lift::{LiftError, lift};
use ktir_superdsc::program::{Program, Segment, TableData};
use triton_frontend::codegen::{ArgSpec, KernelSpec};
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

struct Case {
    name: &'static str,
    file: &'static str,
    entry: &'static str,
    args: Vec<(&'static str, &'static str)>,
    constexprs: Vec<(&'static str, Val)>,
    grid: Vec<i64>,
}

fn kernels_dir() -> String {
    format!("{}/../../../kernels", env!("CARGO_MANIFEST_DIR"))
}

fn lift_case(c: &Case) -> Result<Program, LiftError> {
    let path = format!("{}/{}", kernels_dir(), c.file);
    let src = std::fs::read_to_string(&path).expect("read kernel");
    let mut signature: HashMap<String, ArgSpec> =
        c.args.iter().map(|(n, t)| (n.to_string(), ArgSpec::parse(t).expect("arg type"))).collect();
    for (n, _) in &c.constexprs {
        signature.insert(n.to_string(), ArgSpec::Constexpr);
    }
    let spec = KernelSpec {
        kernel: c.entry.to_string(),
        signature,
        constexprs: c.constexprs.iter().map(|(n, v)| (n.to_string(), v.clone())).collect(),
        file: path,
    };
    let mut tt = triton_frontend::codegen::compile(&src, &spec, Target::spyre()).expect("codegen");
    triton_frontend::opt::make_ttir(&mut tt).expect("make_ttir");
    let mut m = triton_ktir::from_ttir::convert(&tt).expect("from_ttir");
    triton_ktir::make_ktir(&mut m, &c.grid).expect("make_ktir");
    triton_ktir::passes::to_ktir::run(&mut m, &c.grid).expect("to_ktir");
    let f = triton_ktir::passes::to_ktir_emit::lower(&m, Arena::global()).expect("handoff");
    lift(&f)
}

fn i(n: i128) -> Val {
    Val::Int(n)
}

fn f(x: f64) -> Val {
    Val::Float(x)
}

/// The block size the scheduler pages KV in.
const BLOCK_SIZE: i128 = 64;

fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    let case = |name, file, entry, args: &[(&'static str, &'static str)], constexprs: Vec<(&'static str, Val)>, grid: Vec<i64>| Case {
        name,
        file,
        entry,
        args: args.to_vec(),
        constexprs,
        grid,
    };
    let ew = |m: i128, n: i128| {
        vec![
            ("M", i(m)),
            ("N", i(n)),
            ("BLOCK_M", i(m)),
            ("BLOCK_N", i(n)),
            ("N_TOTAL", i(n)),
            ("C_START", i(0)),
            ("N_BLOCKS", i(1)),
            ("TAIL_H", i(0)),
        ]
    };
    let x_o: &[(&str, &str)] = &[("desc_x", "*fp16"), ("desc_o", "*fp16")];
    let a_b_o: &[(&str, &str)] = &[("desc_a", "*fp16"), ("desc_b", "*fp16"), ("desc_o", "*fp16")];
    let x_w_o: &[(&str, &str)] = &[("desc_x", "*fp16"), ("desc_w", "*fp16"), ("desc_o", "*fp16")];
    for entry in ["rmsnorm_fwd", "rmsnorm_oneplus_fwd"] {
        let mut ces = vec![
            ("M", i(1)),
            ("D_MODEL", i(2048)),
            ("BLOCK_M", i(1)),
            ("EPS", f(1e-5)),
            ("INV_D", f(1.0 / 2048.0)),
        ];
        if entry == "rmsnorm_oneplus_fwd" {
            ces.push(("GAIN_OFFSET", f(1.0)));
        }
        v.push(case(entry, "rmsnorm.py", entry, x_w_o, ces, vec![1]));
    }
    let mm = |k: i128, n: i128| {
        vec![
            ("M", i(1)),
            ("K", i(k)),
            ("N", i(n)),
            ("BLOCK_M", i(1)),
            ("BLOCK_K", i(k)),
            ("BLOCK_N", i(n)),
            ("M_TOTAL", i(1)),
        ]
    };
    v.push(case(
        "matmul_fwd",
        "matmul.py",
        "matmul_fwd",
        &[("desc_a", "*fp16"), ("desc_w", "*fp16"), ("desc_o", "*fp16")],
        mm(2048, 2048),
        vec![1],
    ));
    v.push(case(
        "matmul_fp8_fwd",
        "matmul_fp8.py",
        "matmul_fp8_fwd",
        &[("desc_x", "*fp16"), ("desc_w", "*fp8e4nv"), ("desc_ws", "*fp16"), ("desc_o", "*fp16")],
        mm(2048, 2048),
        vec![1],
    ));
    for entry in ["add_fwd", "mul_fwd", "sub_fwd"] {
        v.push(case(entry, "elementwise.py", entry, a_b_o, ew(1, 2048), vec![1]));
    }
    for entry in ["silu_fwd", "gelu_fwd"] {
        v.push(case(entry, "elementwise.py", entry, x_o, ew(1, 2048), vec![1]));
    }
    // A row-blocked elementwise kernel: its block loop is a parallel axis, never a walk.
    let mut blocked = ew(32, 2048);
    blocked[2] = ("BLOCK_M", i(8));
    blocked[6] = ("N_BLOCKS", i(4));
    v.push(case("add_fwd_blocked", "elementwise.py", "add_fwd", a_b_o, blocked, vec![1]));
    v.push(case(
        "silumul_fwd",
        "silumul.py",
        "silumul_fwd",
        &[("desc_g", "*fp16"), ("desc_u", "*fp16"), ("desc_o", "*fp16")],
        ew(1, 2048),
        vec![1],
    ));
    let mut sm = ew(1, 2048);
    sm.push(("SCALE", f(8.0)));
    v.push(case("scalarmul_fwd", "scalarmul.py", "scalarmul_fwd", x_o, sm, vec![1]));
    v.push(case(
        "rope_fwd",
        "rope.py",
        "rope_fwd",
        &[("desc_x", "*fp16"), ("desc_cos", "*fp16"), ("desc_sin", "*fp16"), ("desc_o", "*fp16")],
        vec![("H", i(8)), ("MQ", i(1)), ("HEAD_DIM", i(128)), ("HALF", i(64)), ("Y_TOTAL", i(8)), ("R_START", i(0))],
        vec![1],
    ));
    v.push(case(
        "tanhsoftcap_fwd",
        "tanhsoftcap.py",
        "tanhsoftcap_fwd",
        x_o,
        vec![
            ("M", i(1)),
            ("N", i(8192)),
            ("BLOCK_M", i(1)),
            ("BLOCK_N", i(8192)),
            ("CAP", f(30.0)),
            ("N_BLOCKS", i(1)),
            ("TAIL_H", i(0)),
        ],
        vec![1],
    ));
    v.push(case(
        "rmsnormunit_fwd",
        "rmsnormunit.py",
        "rmsnormunit_fwd",
        x_o,
        vec![("M", i(1)), ("D_MODEL", i(128)), ("BLOCK_M", i(1)), ("EPS", f(1e-5)), ("INV_D", f(1.0 / 128.0))],
        vec![1],
    ));
    v.push(case(
        "scalarwmul_fwd",
        "scalarwmul.py",
        "scalarwmul_fwd",
        &[("desc_x", "*fp16"), ("w", "fp16"), ("desc_o", "*fp16")],
        vec![("M", i(1)), ("N", i(2048))],
        vec![1],
    ));
    v.push(case(
        "reshape_fwd",
        "reshape.py",
        "reshape_fwd",
        &[("desc_src", "*fp16"), ("desc_dst", "*fp16")],
        vec![("R_IN", i(1)), ("C_IN", i(1024)), ("R_OUT", i(2)), ("C_OUT", i(512)), ("N_STICKS", i(16))],
        vec![1],
    ));
    // A copy of 32 rows of 1024 into columns [1024, 2048) of a 2048-wide buffer, in blocks of 8
    // rows; the tail entry with 4 rows past the blocks.
    let copy_args: &[(&str, &str)] = &[("desc_src", "*fp16"), ("desc_dst", "*fp16")];
    let copy = vec![("C_SRC", i(1024)), ("C_DST", i(2048)), ("COL_OFF", i(1024)), ("BLOCK_M", i(8)), ("N_BLOCKS", i(4))];
    v.push(case("blockcopy_fwd", "blockcopy.py", "blockcopy_fwd", copy_args, copy.clone(), vec![1]));
    let mut tail = copy;
    tail.push(("TAIL_H", i(4)));
    v.push(case("blockcopy_tail_fwd", "blockcopy.py", "blockcopy_tail_fwd", copy_args, tail, vec![1]));
    v.push(case(
        "lmlast_fwd",
        "lmlast.py",
        "lmlast_fwd",
        &[("desc_src", "*fp16"), ("desc_dst", "*fp16")],
        vec![("MQ", i(10)), ("HIDDEN", i(2048)), ("W", i(5)), ("S", i(2)), ("N_STICKS", i(32))],
        vec![1],
    ));
    for (entry, gained, oneplus) in [
        ("windowedrmsnorm_gained_fwd", true, false),
        ("windowedrmsnorm_oneplus_fwd", true, true),
        ("windowedrmsnorm_unit_fwd", false, false),
    ] {
        let args: &[(&str, &str)] = if gained {
            &[("desc_x", "*fp16"), ("desc_gamma", "*fp16"), ("desc_o", "*fp16")]
        } else {
            x_o
        };
        let mut ces = vec![("M", i(1)), ("D", i(128)), ("HEADS", i(8)), ("EPS", f(1e-6)), ("INV_D", f(1.0 / 128.0))];
        if oneplus {
            ces.push(("GAIN_OFFSET", f(1.0)));
        }
        v.push(case(entry, "windowedrmsnorm.py", entry, args, ces, vec![1]));
    }
    for (name, mq, blocks, kv_heads) in [
        ("paged_attn_decode", 1i128, 4i128, 8i128),
        ("paged_attn_decode_few_kv_heads", 1, 4, 2),
        ("paged_attn_prefill", 64, 4, 8),
    ] {
        let decode = mq == 1;
        let mut args = vec![
            ("query", "*fp16"),
            ("query_row_index", "*i32"),
            ("k_pages", "*fp16"),
            ("v_pages", "*fp16"),
            ("page_index_table", "*i32"),
        ];
        if decode {
            args.push(("kv_row_pool", "*i32"));
        }
        args.extend([("mask_stack", "*fp16"), ("out", "*fp16")]);
        let mut ces = vec![
            ("NUM_TOKENS", i(mq)),
            ("NUM_HEADS", i(4 * kv_heads)),
            ("NUM_KV_HEADS", i(kv_heads)),
            ("HEAD_SIZE", i(64)),
            ("BLOCK_SIZE", i(BLOCK_SIZE)),
            ("NUM_PAGES", i(16)),
            ("NUM_BLOCKS", i(blocks)),
            ("SCALE", f(0.125)),
        ];
        let (file, entry) = if decode {
            ("page_attn_head_major_decode.py", "page_attn_head_major_decode_kernel")
        } else {
            ces.push(("PADDED_QUERY_LEN", i(mq)));
            ces.push(("BLOCKS_PER_TRIP", i(1)));
            ("page_attn_head_major_prefill.py", "page_attn_head_major_prefill_kernel")
        };
        v.push(case(name, file, entry, &args, ces, vec![1]));
    }
    v.push(case(
        "batched_decode",
        "batched_decode_head_major.py",
        "batched_decode_head_major_kernel",
        &[
            ("query", "*fp16"),
            ("rep_row_ids", "*i32"),
            ("k_pages", "*fp16"),
            ("v_pages", "*fp16"),
            ("chunk_page_ids", "*i32"),
            ("mask_by_chunk", "*fp16"),
            ("out", "*fp16"),
        ],
        vec![
            ("NUM_TOKENS", i(4)),
            ("NUM_SEQS", i(4)),
            ("BLOCKS_PER_CHUNK", i(4)),
            ("NUM_CHUNKS", i(1)),
            ("NUM_KV_HEADS", i(8)),
            ("NUM_QUERIES_PER_KV", i(4)),
            ("BLOCK_SIZE", i(BLOCK_SIZE)),
            ("HEAD_SIZE", i(64)),
            ("NUM_PAGES", i(2048)),
            ("ROWS_PER_PAGE", i(8)),
            ("SCALE", f(0.125)),
        ],
        vec![1],
    ));
    v.push(case(
        "paged_kv_write",
        "reshape_and_cache_head_major.py",
        "reshape_and_cache_head_major_kernel",
        &[
            ("key", "*fp16"),
            ("value", "*fp16"),
            ("k_rows", "*fp16"),
            ("v_rows", "*fp16"),
            ("row_index", "*i32"),
        ],
        vec![("T", i(1)), ("NUM_KV_HEADS", i(8)), ("HEAD_SIZE", i(64)), ("NUM_ROWS", i(4 * 8 * BLOCK_SIZE))],
        vec![1],
    ));
    let h = 128.0f64;
    let moe_ces = |rows: &'static str, m: i128| {
        vec![
            (rows, i(m)),
            ("H", i(128)),
            ("I", i(64)),
            ("E", i(64)),
            ("K", i(8)),
            ("EPS", f(1e-6)),
            ("INV_H", f(1.0 / h)),
            ("TEMP", f(h.powf(-0.5))),
            ("INV_QMAX", f(1.0 / 448.0)),
            ("SCALE_FLOOR", f(1.0 / 448.0 / 448.0)),
        ]
    };
    let moe_head: [(&str, &str); 12] = [
        ("desc_xr", "*fp16"),
        ("desc_rg", "*fp16"),
        ("desc_rw", "*fp16"),
        ("desc_pes", "*fp16"),
        ("desc_xe", "*fp16"),
        ("desc_wg", "*fp8e4nv"),
        ("desc_sg", "*fp16"),
        ("desc_wu", "*fp8e4nv"),
        ("desc_su", "*fp16"),
        ("desc_wd", "*fp8e4nv"),
        ("desc_sd", "*fp16"),
        ("desc_o", "*fp16"),
    ];
    let mut moe_args = moe_head.to_vec();
    moe_args.extend([("desc_ids", "*i32"), ("desc_rs", "*fp16"), ("desc_act", "*fp16")]);
    v.push(case("moe_fwd", "moe.py", "moe_fwd", &moe_args, moe_ces("M", 1), vec![1]));
    let mut prefill_args = moe_head.to_vec();
    prefill_args.push(("desc_dense", "*fp16"));
    v.push(case("moe_prefill_fwd", "moe_prefill.py", "moe_prefill_fwd", &prefill_args, moe_ces("T", 16), vec![1]));
    let sample_args: &[(&str, &str)] = &[
        ("logits_ptr", "*fp16"),
        ("slo_ptr", "*i32"),
        ("shi_ptr", "*i32"),
        ("stp_ptr", "*i32"),
        ("sgate_ptr", "*i32"),
        ("invt_ptr", "*fp32"),
        ("bramp_ptr", "*fp32"),
        ("lramp_ptr", "*fp32"),
        ("binc_ptr", "*fp32"),
        ("linc_ptr", "*fp32"),
        ("ones_ptr", "*fp32"),
        ("sixtyfour_ptr", "*fp32"),
        ("padoff_ptr", "*fp32"),
        ("gramp_ptr", "*fp32"),
        ("ginc_ptr", "*fp32"),
        ("hc1_ptr", "*i32"),
        ("hc2_ptr", "*i32"),
        ("hc3_ptr", "*i32"),
        ("c256_ptr", "*i32"),
        ("lane_ptr", "*fp32"),
        ("fm8_ptr", "*fp32"),
        ("ftop_ptr", "*fp32"),
        ("fzero_ptr", "*fp32"),
        ("fhalf_ptr", "*fp32"),
        ("fm22_ptr", "*fp32"),
        ("cmin_ptr", "*fp16"),
        ("idx_ptr", "*i32"),
    ];
    for (name, b) in [("sample_fwd_b1", 1i128), ("sample_fwd_b2", 2)] {
        v.push(case(
            name,
            "sample_fwd.py",
            "sample_fwd",
            sample_args,
            vec![("V", i(49155)), ("PAD_V", i(65536)), ("NB", i(1024)), ("BLOCK", i(64)), ("B", i(b))],
            vec![b as i64],
        ));
    }
    v
}

/// What each case does today: it lifts (with its walk count), or the error it stops at.
#[derive(Debug, PartialEq)]
enum Status {
    Lifts { walks: usize },
    Stops(String),
}

/// THE RED LIST, and the green one: every case's state. Changing a line here is a claim the
/// test checks.
const EXPECTED: &[(&str, &str)] = &[
    ("rmsnorm_fwd", "lifts, 0 walk(s)"),
    ("rmsnorm_oneplus_fwd", "lifts, 0 walk(s)"),
    ("matmul_fwd", "lifts, 0 walk(s)"),
    ("matmul_fp8_fwd", "lifts, 0 walk(s)"),
    ("add_fwd", "lifts, 0 walk(s)"),
    ("mul_fwd", "lifts, 0 walk(s)"),
    ("sub_fwd", "lifts, 0 walk(s)"),
    ("silu_fwd", "lifts, 0 walk(s)"),
    ("gelu_fwd", "lifts, 0 walk(s)"),
    ("add_fwd_blocked", "lifts, 0 walk(s)"),
    ("silumul_fwd", "lifts, 0 walk(s)"),
    ("scalarmul_fwd", "lifts, 0 walk(s)"),
    ("rope_fwd", "lifts, 0 walk(s)"),
    ("tanhsoftcap_fwd", "lifts, 0 walk(s)"),
    ("rmsnormunit_fwd", "lifts, 0 walk(s)"),
    ("scalarwmul_fwd", "lifts, 0 walk(s)"),
    ("reshape_fwd", "lifts, 0 walk(s)"),
    ("blockcopy_fwd", "lifts, 0 walk(s)"),
    ("blockcopy_tail_fwd", "lifts, 0 walk(s)"),
    ("lmlast_fwd", "lifts, 0 walk(s)"),
    ("windowedrmsnorm_gained_fwd", "lifts, 0 walk(s)"),
    ("windowedrmsnorm_oneplus_fwd", "lifts, 0 walk(s)"),
    ("windowedrmsnorm_unit_fwd", "lifts, 0 walk(s)"),
    ("paged_attn_decode", "lifts, 1 walk(s)"),
    // RED: `tl.min` over int32 row ids; the device reduces fp16/fp32 only
    // (summeanmaxexx2*.ddl), so this needs the int→fp32, reduce, fp32→int legalization.
    ("paged_attn_decode_few_kv_heads", "stops: Op(ArithMinsi)"),
    ("paged_attn_prefill", "lifts, 1 walk(s)"),
    ("batched_decode", "lifts, 0 walk(s)"),
    ("paged_kv_write", "lifts, 0 walk(s)"),
    // The MoE kernels: the router and three contractions each (tests/moe.rs: four contractions
    // at any token, prompt or expert count, and numerically the host's) — moe_fwd's routed
    // experts' slabs gathered by id, moe_prefill_fwd's over all E experts.
    ("moe_fwd", "lifts, 0 walk(s)"),
    ("moe_prefill_fwd", "lifts, 0 walk(s)"),
    // The sampler lifts. Lifting is not the draw being right: the numeric check of the
    // random AND greedy draws against an exact host computation is its own gate. The grid
    // ([B], one program per request row) is one outer axis (tests/grid.rs). No production
    // kernel runs a grid wider than the card's 32 cores: the sampler's grid is the batch
    // rung, whose widest is 32 (the old crate's `PagedKvPool::WIDEST_BATCH_RUNG`, a rung
    // fact the bake carries).
    ("sample_fwd_b1", "lifts, 0 walk(s)"),
    ("sample_fwd_b2", "lifts, 0 walk(s)"),
];

fn status(r: &Result<Program, LiftError>) -> Status {
    match r {
        Ok(p) => Status::Lifts {
            walks: p.segments.iter().filter(|s| matches!(s, Segment::Walk(_))).count(),
        },
        Err(e) => Status::Stops(format!("{e:?}")),
    }
}

/// Every `@triton.jit` entry point (a name not starting with `_`) the kernel files define.
fn entry_points() -> BTreeSet<(String, String)> {
    let mut out = BTreeSet::new();
    for e in std::fs::read_dir(kernels_dir()).expect("kernels dir") {
        let path = e.expect("entry").path();
        if path.extension().is_none_or(|x| x != "py") {
            continue;
        }
        let file = path.file_name().expect("name").to_string_lossy().to_string();
        let src = std::fs::read_to_string(&path).expect("read");
        let mut jit = false;
        for line in src.lines() {
            if line.starts_with("@triton.jit") {
                jit = true;
            } else if let Some(rest) = line.strip_prefix("def ") {
                let name = rest.split('(').next().unwrap_or("").to_string();
                if jit && !name.starts_with('_') {
                    out.insert((file.clone(), name));
                }
                jit = false;
            }
        }
    }
    out
}

#[test]
fn every_kernel_lifts_or_is_on_the_red_list() {
    let cases = cases();
    let covered: BTreeSet<(String, String)> =
        cases.iter().map(|c| (c.file.to_string(), c.entry.to_string())).collect();
    let missing: Vec<_> = entry_points().difference(&covered).cloned().collect();
    assert!(missing.is_empty(), "kernel entry points with no case: {missing:?}");

    let mut report = Vec::new();
    let mut wrong = Vec::new();
    for c in &cases {
        let lifted = lift_case(c);
        // Every constant table a kernel states is a ramp (each `tl.arange`), stated by its
        // form: no kernel stages a table element by element.
        if let Ok(p) = &lifted {
            assert!(
                p.tables.iter().all(|t| matches!(t.data, TableData::Ramp { .. })),
                "{}: a table stated element by element",
                c.name
            );
        }
        let got = status(&lifted);
        let line = match &got {
            Status::Lifts { walks } => format!("lifts, {walks} walk(s)"),
            Status::Stops(e) => format!("stops: {e}"),
        };
        report.push(format!("{:<34} {line}", c.name));
        let want = EXPECTED.iter().find(|(n, _)| *n == c.name).map(|(_, s)| *s);
        if want != Some(line.as_str()) {
            wrong.push(format!("{}: expected {want:?}, got {line:?}", c.name));
        }
    }
    assert!(wrong.is_empty(), "\n{}\n\nmismatches:\n{}", report.join("\n"), wrong.join("\n"));
}
