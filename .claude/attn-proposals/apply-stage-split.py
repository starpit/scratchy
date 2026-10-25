import sys

p = "crates/targets/spyre/src/lower_subtile_tape_to_superdsc.rs"
lines = open(p).read().split("\n")
assert lines[8073] == "    // ── from here down this is main's block, verbatim ──", lines[8073]
assert lines[8125] == "    crate::pc::bundle_line(fp, ops.len(), sdsc_ops.len(), staged);", lines[8125]
assert lines[8126] == "    Ok(groups)", lines[8126]

new = '''    // ── from here down this is main's block, verbatim — EXCEPT THE SPLIT: the staging moved to
    //    `StageHandle::submit`, the commit half the macro's ladder runs after the total-budget
    //    decision. What stays here is everything that decision needs: the launch index, the launch
    //    pairs, the trips, and the rendered json the submit will write. ──
    let shifts = {
        let _pc = crate::pc::Pc::start("launch");
        launch_index(&sdsc_ops, fold)
    };
    // Every launch pair, IN LAUNCH ORDER — computable at render because a group's identity is its
    // CONTENT (`GroupId { fp, group }`), not its compiled artifact. The shift-disagreement fence
    // is the same fence it always was; only the point it runs moved, ahead of the staging it
    // guards. The adopt/memo pass inside the submit hands this SAME list back, adopt-hits
    // included, exactly as the staging loop built it.
    let mut groups: Vec<(bundle::KvShifts, crate::superdsc_bake::GroupId)> = Vec::new();
    for g in &rendered {
        let id = crate::superdsc_bake::GroupId {
            fp: fp.to_string(),
            group: g.group,
        };
        let kv = *shifts.get(g.group as usize).ok_or_else(|| {
            EmitError::Fatal(format!(
                "{fp}: group {} has dxp input but no entry in the launch index ({} entries) — \\
                 `render_dxp_input` and `launch_index` disagree about the group partition",
                g.group,
                shifts.len(),
            ))
        })?;
        groups.push((kv, id));
    }
    Ok(StageHandle {
        fp: fp.to_string(),
        groups,
        rendered,
        trips,
        programs: ops.len(),
        descriptors: sdsc_ops.len(),
    })
}'''

lines[8073:8128] = new.split("\n")
open(p, "w").write("\n".join(lines))
print("tail ok")
