"""Materialize four iterations directly in TTIR; no reliance on a frontend pass."""

import re

from common import ROOT, ir_path, manifest, save_json, sha256


def unroll_four(text):
    if "scf.for" not in text:
        return text
    loop = re.search(
        r"(?P<head>    (?P<result>%\w+):2 = scf.for %start_n = %c0_i32 to %N step "
        r"(?P<step>%c(?P<width>\d+)_i32) iter_args\(%m = %\w+, (?P<z>%\w+) = %\w+\) "
        r"-> \(tensor<[^>]+>, tensor<[^>]+>\)  : i32 \{\n)"
        r"(?P<body>.*?)    }\n",
        text,
        re.S,
    )
    assert loop
    width = int(loop["width"])
    body = loop["body"]
    yielded = re.search(r"      scf.yield (%\w+), (%\w+) : (.*?)\n", body)
    assert yielded
    body = body[: yielded.start()]
    definitions = set(re.findall(r"^      (%\w+) =", body, re.M))
    chunks = []
    carried = ["%m", loop["z"]]
    for index in range(4):
        mapping = {name: f"%u{index}_{name[1:]}" for name in definitions}
        mapping.update({"%m": carried[0], loop["z"]: carried[1]})
        if index:
            chunks.append(
                f"      %u{index}_offset = arith.constant {index*width} : i32\n"
                f"      %u{index}_index = arith.addi %start_n, %u{index}_offset : i32\n"
            )
            mapping["%start_n"] = f"%u{index}_index"
        chunks.append(re.sub(r"%\w+", lambda m: mapping.get(m[0], m[0]), body))
        carried = [mapping[yielded[1]], mapping[yielded[2]]]
    chunks.append(f"      scf.yield {carried[0]}, {carried[1]} : {yielded[3]}\n")
    result = (
        text[: loop.start()]
        + loop["head"]
        + "".join(chunks)
        + "    }\n"
        + text[loop.end() :]
    )
    return result.replace(
        f'{loop["step"]} = arith.constant {width} :',
        f'{loop["step"]} = arith.constant {4*width} :',
    )


def main():
    groups = {}
    for group in manifest()["groups"]:
        parent = ir_path(group, "r2_single_exp")
        path = ir_path(group, "r2_explicit4")
        with path.open("x") as stream:
            stream.write(unroll_four(parent.read_text()))
        groups[group] = dict(sha256=sha256(path), parent_sha256=sha256(parent))
    save_json(
        ROOT / "results/runs/r2_setup/candidate_explicit4.json",
        dict(
            groups=groups,
            generator_sha256=sha256(__file__),
            hypothesis="Materialized unroll versus the ignored loop hint; original loads stay masked",
        ),
    )


if __name__ == "__main__":
    main()
