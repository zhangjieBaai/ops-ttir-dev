"""Direct TTIR output-row subdivision and scalar pointer rematerialization."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from common import digest, group, save  # noqa: E402


def immutable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == text, f"Immutable candidate differs: {path}"
    else:
        path.write_text(text)


def subdivide(text, width, head):
    lines = text.splitlines(keepends=True)
    start = next(
        i for i, line in enumerate(lines) if "    %offs_k = tt.make_range" in line
    )
    end = next(i for i, line in enumerate(lines) if "    tt.return" in line)
    # Accumulator and dot-zero constants are key-row-shaped, not head-shaped.
    for i in range(start):
        if "arith.constant dense" in lines[i]:
            lines[i] = lines[i].replace("tensor<128x", f"tensor<{width}x")
    body = []
    for original in lines[start:end]:
        line = original.replace("tensor<128x", f"tensor<{width}x")
        stripped = original.strip()
        # These values have head-D as their leading dimension and do not shrink.
        if re.match(r"%(offs_k|qT|qT_ptrs\w*|do|do_ptrs\w*) =", stripped):
            line = original
        if " = tt.trans " in original:
            line = original
        if stripped.startswith("%start_n ="):
            line = original.replace("%start_n =", "%r2_block_start =")
            line += "    %start_n = arith.addi %r2_block_start, %r2_slice : i32\n"
            line += (
                f"    %r2_key_range = tt.make_range {{end = {width} : i32, start = 0 : i32}}"
                f" : tensor<{width}xi32>\n"
            )
        elif stripped.startswith("%offs_n"):
            line = line.replace("tensor<128xi32>", f"tensor<{width}xi32>")
            line = line.replace(", %offs_k :", ", %r2_key_range :")
        elif re.match(r"%k = tt.expand_dims", stripped):
            line = line.replace("tensor<128xi32>", f"tensor<{width}xi32>")
        if " = tt.dot " in original:
            if stripped.startswith("%qkT") or stripped.startswith("%dpT"):
                # K/V rows shrink, Q/dO transpose head dimension does not.
                line = re.sub(
                    rf"\* tensor<{width}x(16|32)xf16>",
                    rf"* tensor<{head}x\1xf16>",
                    line,
                )
        if "scf.for" in line or "scf.yield" in line:
            line = re.sub(
                rf"tensor<{width}x(16|32)x!tt.ptr<f16>>",
                rf"tensor<{head}x\1x!tt.ptr<f16>>",
                line,
            )
        # Causal query iteration retains the original 128-key block boundaries.
        if not stripped.startswith(("%start_n =", "%offs_n")):
            line = re.sub(r"%start_n\b", "%r2_block_start", line)
        body.append("  " + line.replace("\n    ", "\n      "))
    prefix = (
        f"    %r2_width = arith.constant {width} : i32\n"
        "    scf.for %r2_slice = %c0_i32 to %c128_i32 step %r2_width {\n"
    )
    return (
        "".join(lines[:start])
        + prefix
        + "".join(body)
        + "    }\n"
        + "".join(lines[end:])
    )


def remat(text):
    lines = text.splitlines(keepends=True)
    result = []
    for line in lines:
        if "scf.for" not in line or "iter_args" not in line:
            result.append(line)
            continue
        m = re.search(
            r"(%qT_ptrs_\d+) = (%qT_ptrs_\d+), (%do_ptrs_\d+) = (%do_ptrs_\d+), (%curr_m_\d+) = (%\w+)",
            line,
        )
        assert m, line
        qvar, qinitial, ovar, oinitial, current, initial = m.groups()
        types = re.findall(r"tensor<\d+x\d+x!tt.ptr<f16>>", line)
        assert len(types) == 2
        qty, oty = types
        line = line.replace(f"{qvar} = {qinitial}, {ovar} = {oinitial}, ", "")
        line = line.replace(f", {qty}, {oty}", "")
        line = line.replace(":5 = scf.for", ":3 = scf.for")
        result.append(line)
        indent = line[: len(line) - len(line.lstrip())] + "  "
        qshape = qty.replace("!tt.ptr<f16>", "i32")
        oshape = oty.replace("!tt.ptr<f16>", "i32")
        result.extend(
            [
                f"{indent}%r2_delta = arith.subi {current}, {initial} : i32\n",
                f"{indent}%r2_offset = arith.muli %r2_delta, %stride_tok : i32\n",
                f"{indent}%r2_qoffset = tt.splat %r2_offset : i32 -> {qshape}\n",
                f"{indent}{qvar} = tt.addptr {qinitial}, %r2_qoffset : {qty}, {qshape}\n",
                f"{indent}%r2_ooffset = tt.splat %r2_offset : i32 -> {oshape}\n",
                f"{indent}{ovar} = tt.addptr {oinitial}, %r2_ooffset : {oty}, {oshape}\n",
            ]
        )
    text = "".join(result)
    # Pointer yields are no longer part of the region interface; now-dead advances
    # remain for the native downstream DCE, so no upstream pass is manually run.
    text = re.sub(
        r"(scf.yield %\w+, %\w+), %qT_ptrs_\d+, %do_ptrs_\d+(, %curr_m_\d+)",
        r"\1\2",
        text,
    )
    text = re.sub(
        r"(scf.yield[^\n]+ : tensor<[^>]+>, tensor<[^>]+>), tensor<\d+x\d+x!tt.ptr<f16>>, tensor<\d+x\d+x!tt.ptr<f16>>",
        r"\1",
        text,
    )
    return text


def main():
    manifest = {}
    for d in (64, 128):
        for c in (False, True):
            g = group(d, c)
            folder = ROOT / "ir" / g
            baseline = (ROOT.parent / "ir" / g / "baseline.ttir").read_text()
            parent_name = "r1_dv_late" if d == 64 else "baseline"
            parent_path = ROOT.parent / "ir" / g / f"{parent_name}.ttir"
            parent = parent_path.read_text()
            immutable(folder / "baseline.ttir", baseline)
            immutable(folder / "parent.ttir", parent)
            variants = {
                "rows64_v1": subdivide(parent, 64, d),
                "rows32_v1": subdivide(parent, 32, d),
                "remat_ptrs_v1": remat(parent),
            }
            variants["rows64_remat_v1"] = remat(variants["rows64_v1"])
            variants["rows32_remat_v1"] = remat(variants["rows32_v1"])
            manifest[g] = {
                "parent_name": parent_name,
                "parent_sha256": digest(parent),
                "candidates": {},
            }
            for name, content in variants.items():
                path = folder / f"{name}.ttir"
                immutable(path, content)
                manifest[g]["candidates"][name] = digest(content)
    save(ROOT / "manifest.json", manifest)


if __name__ == "__main__":
    main()
