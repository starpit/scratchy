import sys

p = "crates/targets/spyre/src/lower_subtile_tape_to_ktir.rs"
src = open(p).read()
lines = src.split("\n")

# ── 1. module doc (4561-4563) + name (4565), 1-based ──
assert lines[4560].startswith("/// ⭐ THE TRIP-WALL SPLIT"), lines[4560]
assert lines[4564] == "mod trip_wall_split_tests {", lines[4564]
lines[4560] = "/// ⭐ THE TRIP-WALL CENSUS'S OWN TESTS — the over-wall body flows to the mint"
lines[4561] = "/// whole (nothing moves), the under-wall body passes through, and the counting pass"
lines[4562] = "/// leaves the roll's layout untouched, each over the same hand-stated three-program"
lines[4563] = "/// roll (the `zz_` roster's law: real op kinds, hand-stated bindings, output last)."
lines[4564] = "mod trip_wall_census_tests {"

# ── 2. T1 rewrite: replace lines 5347-5454 (1-based inclusive) ──
assert lines[5346] == "    #[test]", lines[5346]
assert "a_body_over_the_trip_wall_splits_at_the_residual_seam" in lines[5347]
assert lines[5453] == "    }", lines[5453]
new_t1 = '''    #[test]
    fn a_body_over_the_trip_wall_flows_to_the_mint_whole() {
        let wall = crate::superdsc_bake::BakeBudget::MAX_BUNDLE_TRIPS;
        let rolled = rung_roll();
        let whole = trips(&rolled.bodies[0], &rolled.layout);
        assert!(
            whole > wall,
            "the rung fixture must sit OVER the wall by the door's own count — got {whole} \\
             trips against the {wall} wall"
        );
        census_bodies_over_the_trip_wall(&rolled);
        // ── NOTHING MOVES — the census is the log's attribution: the over-wall body flows to
        //    the mint whole and its `BakeBudget` refusal fires at the mint's own site.
        assert_eq!(rolled.bodies.len(), 1, "the roll keeps its one body class");
        assert_eq!(
            rolled.bodies[0].ops.len(),
            8,
            "the body keeps all eight ops — four attention contractions, the seam add, three \\
             MLP contractions"
        );
        assert_eq!(
            rolled.launches,
            vec![(0u8, 0u32), (0u8, 1u32)],
            "the launch sequence is untouched"
        );
        // rung_weight_shift() = the layer-0 packed weight total (4,198,400 + 1,049,600 +
        // 1,049,600 + 4,198,400 + 16,793,600 + 16,793,600 + 16,781,312), the layer-1 shift.
        assert_eq!(
            rolled.launch_w,
            vec![(0u8, 0u64), (0u8, rung_weight_shift())],
            "the launch rows are untouched"
        );
        assert_eq!(
            rolled.launch_kv,
            vec![0u64, 0],
            "the launch kv rows are untouched"
        );
    }'''
lines[5346:5454] = new_t1.split("\n")

# ── 3. delete the seam-leak test (now shifted: find it by content) ──
i = next(k for k, l in enumerate(lines) if "a_body_whose_seam_leaks_a_staging_refuses_the_split" in l)
# back up to its #[test] line
start = i - 1
assert lines[start] == "    #[test]", lines[start]
# forward to its closing brace
end = next(k for k in range(i, len(lines)) if lines[k] == "    }")
# delete #[test] .. closing brace inclusive, plus the blank line after
assert lines[end + 1] == "", repr(lines[end + 1])
del lines[start : end + 2]

src = "\n".join(lines)

# ── 4. unique string replaces, immune to shifts ──
def rep(old, new):
    global src
    assert src.count(old) == 1, (old, src.count(old))
    src = src.replace(old, new)

# under-wall test: drop mut + plain call
rep("        let mut rolled = seam_roll();", "        let rolled = seam_roll();")
rep(
    '        split_bodies_at_trip_wall(&mut rolled).expect("the no-op path returns Ok");',
    "        census_bodies_over_the_trip_wall(&rolled);",
)
# T5: doc ref, mut, call, and the bodies-count assert
rep(
    "    /// (see [`split_bodies_at_trip_wall`]), so the placements, the synth allocator and the",
    "    /// (see [`census_bodies_over_the_trip_wall`]), so the placements, the synth allocator and",
)
rep(
    '        split_bodies_at_trip_wall(&mut rolled).expect("the legal seam splits");',
    "        census_bodies_over_the_trip_wall(&rolled);",
)
rep(
    '''        assert_eq!(
            rolled.bodies.len(),
            2,
            "the over-wall body split — a refusal here means the fixture rotted, not the pin"
        );''',
    '''        assert_eq!(
            rolled.bodies.len(),
            1,
            "the over-wall body flows to the mint whole — nothing moves; a refusal here means \\
             the fixture rotted, not the pin"
        );''',
)

open(p, "w").write(src)
print("ok")
