"""Deterministic structural edits to frozen TTIR only; never imports a Python kernel."""

import re

from common import ROOT, digest, save


def move_loop_ops(text, mode):
    # Frozen IR uses a straight-line loop body; SSA names repeat per lexical scope.
    lines = text.splitlines(keepends=True)
    spans = []
    for i, line in enumerate(lines):
        if " = scf.for " in line:
            j = next(j for j in range(i + 1, len(lines)) if "scf.yield" in lines[j])
            spans.append((i, j))
    for i, j in reversed(spans):
        body = lines[i + 1 : j]
        if mode == "delta_early":
            selected = [x for x in body if re.match(r"\s+%Di(?:_\d+)? = ", x)]
            body = [x for x in body if x not in selected]
            pos = next(k for k, x in enumerate(body) if "%qkT = tt.dot" in x)
        elif mode == "dv_late":
            selected = [
                x
                for x in body
                if "%ppT = arith.truncf" in x or re.match(r"\s+%dv_\d+ = tt.dot", x)
            ]
            body = [x for x in body if x not in selected]
            pos = (
                next(
                    k for k, x in enumerate(body) if re.match(r"\s+%dk_\d+ = tt.dot", x)
                )
                + 1
            )
        elif mode == "dp_early":
            selected = [
                x
                for x in body
                if "%do = tt.load" in x or re.match(r"\s+%dpT(?:_\d+)? = ", x)
            ]
            body = [x for x in body if x not in selected]
            pos = next(k for k, x in enumerate(body) if "%qkT = tt.dot" in x)
        else:
            raise ValueError(mode)
        assert selected
        body[pos:pos] = selected
        lines[i + 1 : j] = body
    return "".join(lines)


def fold_heads(text):
    # Contract: contiguous B,H,N,D; stride_z = H * stride_h, all offsets < 2^31.
    pattern = r"    %adj = arith.remsi.*?    (?P<result>%adj_\d+) = arith.addi[^\n]+\n"
    match = re.search(pattern, text, re.S)
    assert match
    return (
        text[: match.start()]
        + f"    {match['result']} = arith.muli %bhid, %stride_h : i32\n"
        + text[match.end() :]
    )


def main():
    manifest = {}
    for path in sorted((ROOT / "ir").glob("*/baseline.ttir")):
        text = path.read_text()
        variants = {
            "r1_unsigned": text.replace("arith.divsi", "arith.divui").replace(
                "arith.remsi", "arith.remui"
            ),
            "r1_delta_early": move_loop_ops(text, "delta_early"),
            "r1_dv_late": move_loop_ops(text, "dv_late"),
            "r1_dp_early": move_loop_ops(text, "dp_early"),
            "r1_load_cg": re.sub(r"(tt.load %\w+) :", r"\1 cacheModifier = cg :", text),
            "r1_fold_heads": fold_heads(text),
            "r1_dv_late_unsigned": move_loop_ops(text, "dv_late")
            .replace("arith.divsi", "arith.divui")
            .replace("arith.remsi", "arith.remui"),
        }
        manifest[path.parent.name] = {}
        for name, content in variants.items():
            output = path.with_name(name + ".ttir")
            if output.exists():
                assert output.read_text() == content, "Immutable candidate changed"
            else:
                output.write_text(content)
            manifest[path.parent.name][name] = dict(
                parent_sha256=digest(path), sha256=digest(output)
            )
    save(ROOT / "candidates.json", manifest)


if __name__ == "__main__":
    main()
