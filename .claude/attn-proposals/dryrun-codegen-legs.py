import re, shutil, sys

SRC = "crates/compiler/macros/src/codegen.rs"
COPY = ".claude/attn-proposals/codegen-dryrun.rs"
shutil.copyfile(SRC, COPY)

for leg in ["apply-codegen-leg.py", "apply-codegen-leg2.py", "apply-codegen-leg3.py"]:
    code = open(f".claude/attn-proposals/{leg}").read()
    code = code.replace(
        'p = "crates/compiler/macros/src/codegen.rs"',
        f'p = "{COPY}"',
    )
    # drop the trailing write so the dry run leaves the copy as the chain input? No —
    # chain must persist: keep the write, it writes the copy.
    g = {"__name__": "__main__"}
    exec(compile(code, leg, "exec"), g)
    print(leg, "DRY-RUN OK")
print("all anchors matched; chained output in", COPY)
