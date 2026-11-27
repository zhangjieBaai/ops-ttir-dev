"""Three bounded rounds of direct TTIR transformations, preserving old evidence."""

import argparse
import importlib.util
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
OP = BASE.parent
GROUPS = ["d64_c0", "d64_c1", "d128_c0", "d128_c1"]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def immutable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == text, path
    else:
        path.write_text(text)


def remat(text):
    lines = text.splitlines(keepends=True)
    result = []
    for line in lines:
        if "scf.for" not in line or "iter_args" not in line:
            result.append(line)
            continue
        m = re.search(
            r"(%qT_ptrs_\w+) = (%qT_ptrs_\w+), (%do_ptrs_\w+) = (%do_ptrs_\w+), (%curr_m_\w+) = (%\w+)",
            line,
        )
        if not m:
            result.append(line)
            continue
        qvar, qinitial, ovar, oinitial, current, initial = m.groups()
        qty, oty = re.findall(r"tensor<\d+x\d+x!tt.ptr<f16>>", line)
        line = line.replace(f"{qvar} = {qinitial}, {ovar} = {oinitial}, ", "")
        line = line.replace(f", {qty}, {oty}", "").replace(
            ":5 = scf.for", ":3 = scf.for"
        )
        result.append(line)
        indent = line[: len(line) - len(line.lstrip())] + "  "
        qs, os = qty.replace("!tt.ptr<f16>", "i32"), oty.replace("!tt.ptr<f16>", "i32")
        result += [
            f"{indent}%mr_delta = arith.subi {current}, {initial} : i32\n",
            f"{indent}%mr_offset = arith.muli %mr_delta, %stride_tok : i32\n",
            f"{indent}%mr_qoffset = tt.splat %mr_offset : i32 -> {qs}\n",
            f"{indent}{qvar} = tt.addptr {qinitial}, %mr_qoffset : {qty}, {qs}\n",
            f"{indent}%mr_ooffset = tt.splat %mr_offset : i32 -> {os}\n",
            f"{indent}{ovar} = tt.addptr {oinitial}, %mr_ooffset : {oty}, {os}\n",
        ]
    text = "".join(result)
    text = re.sub(
        r"(scf.yield %\w+, %\w+), %qT_ptrs_\w+, %do_ptrs_\w+(, %curr_m_\w+)",
        r"\1\2",
        text,
    )
    text = re.sub(
        r"(scf.yield[^\n]+ : tensor<[^>]+>, tensor<[^>]+>), tensor<\d+x\d+x!tt.ptr<f16>>, tensor<\d+x\d+x!tt.ptr<f16>>",
        r"\1",
        text,
    )
    return text


def outer_unroll(text):
    if "    scf.for %r2_slice =" not in text:
        return text
    begin = text.index("    scf.for %r2_slice =")
    body_start = text.index("\n", begin) + 1
    end = text.index("    }\n    tt.return", body_start)
    body = text[body_start:end]
    defined = set(re.findall(r"(%\w+)(?::\d+)?\s*=", body))
    output = []
    for index, offset in enumerate((0, 64)):
        output.append(f"    %mr_fixed_{index} = arith.constant {offset} : i32\n")
        mapping = {old: f"%mro{index}_{old[1:]}" for old in defined}
        mapping["%r2_slice"] = f"%mr_fixed_{index}"
        copy = re.sub(r"%\w+", lambda m: mapping.get(m[0], m[0]), body)
        output.append(
            "\n".join(
                line[2:] if line.startswith("  ") else line
                for line in copy.splitlines()
            )
            + "\n"
        )
    return text[:begin] + "".join(output) + text[end + 6 :]


def dp_early(text):
    lines = text.splitlines(keepends=True)
    spans = []
    for i, line in enumerate(lines):
        if " = scf.for " in line and "iter_args" in line:
            end = next(j for j in range(i + 1, len(lines)) if "scf.yield" in lines[j])
            spans.append((i, end))
    for i, j in reversed(spans):
        body = lines[i + 1 : j]
        selected = [
            line
            for line in body
            if "%do = tt.load" in line or re.match(r"\s+%dpT(?:_\d+)? = ", line)
        ]
        assert selected
        body = [line for line in body if line not in selected]
        pos = next(k for k, line in enumerate(body) if "%qkT = tt.dot" in line)
        body[pos:pos] = selected
        lines[i + 1 : j] = body
    return "".join(lines)


def query_pair(text):
    lines = text.splitlines(keepends=True)
    loops = []
    for i, line in enumerate(lines):
        if " = scf.for " in line and "iter_args" in line:
            end = next(j for j in range(i + 1, len(lines)) if "scf.yield" in lines[j])
            loops.append((i, end))
    for start, end in reversed(loops):
        header = lines[start]
        iv = re.search(r"scf.for (%\w+)", header)[1]
        body = "".join(lines[start + 1 : end])
        assert not re.search(re.escape(iv) + r"\b", body)
        args = header.split("iter_args(")[1].split(")")[0]
        pairs = re.findall(r"(%\w+) = (%\w+(?:#\d+)?)", args)
        yielded = re.findall(r"%\w+(?:#\d+)?", lines[end].split(" : ")[0])
        assert len(pairs) == len(yielded)
        defs = set(re.findall(r"(%\w+)(?::\d+)? = ", body))
        expanded = []
        current = {name: name for name, _ in pairs}
        for pass_id in range(2):
            mapping = {name: f"%mrp{pass_id}_{name[1:]}" for name in defs}
            mapping.update(current)
            expanded.append(
                re.sub(r"%\w+(?:#\d+)?", lambda m: mapping.get(m[0], m[0]), body)
            )
            current = {
                name: mapping.get(value, value)
                for (name, _), value in zip(pairs, yielded)
            }
        final_yield = re.sub(
            r"%\w+(?:#\d+)?", lambda m: mapping.get(m[0], m[0]), lines[end]
        )
        lines[start : end + 1] = [
            header.replace("step %c1_i32", "step %mr_pair_step"),
            *expanded,
            final_yield,
        ]
    text = "".join(lines)
    marker = "    %c1_i32 ="
    assert marker in text
    return text.replace(
        marker, "    %mr_pair_step = arith.constant 2 : i32\n" + marker, 1
    )


def move_delta(text):
    lines = text.splitlines(keepends=True)
    spans = []
    for i, line in enumerate(lines):
        if " = scf.for " in line and "iter_args" in line:
            j = next(j for j in range(i + 1, len(lines)) if "scf.yield" in lines[j])
            spans.append((i, j))
    for i, j in reversed(spans):
        body = lines[i + 1 : j]
        # Pair-unrolled bodies use mrp prefixes; retain each half's dependencies.
        starts = [
            k
            for k, line in enumerate(body)
            if re.search(r"%[\w]*qkT[\w]* = tt.dot", line)
        ]
        for index, start in reversed(list(enumerate(starts))):
            stop = starts[index + 1] if index + 1 < len(starts) else len(body)
            selected = [
                line
                for line in body[start:stop]
                if re.match(r"\s+%(?:(?:mrp|mro)\d+_)*Di(?:_\w+)? =", line)
            ]
            if not selected:
                continue
            # Di addresses depend only on offsets built before this qk dot.
            segment = body[start:stop]
            body[start:stop] = selected + [
                line for line in segment if line not in selected
            ]
        lines[i + 1 : j] = body
    return "".join(lines)


def qdo_policy(text, policy):
    lines = []
    for line in text.splitlines(keepends=True):
        if re.match(r"\s+%(?:(?:mrp|mro)\d+_)*(?:qT|do)(?:_\w+)? = tt.load", line):
            line = re.sub(r" cacheModifier = \w+", "", line)
            line = re.sub(r" evictionPolicy = \w+", "", line)
            line = line.replace(" : tensor", f" {policy} : tensor", 1)
        lines.append(line)
    return "".join(lines)


def main(round_name):
    rd = BASE / round_name
    assert round_name in ("r3", "r4", "r5")
    previous = None if round_name == "r3" else BASE / f"r{int(round_name[1:])-1}"
    manifest = {}
    for group in GROUPS:
        baseline = (OP / "ir" / group / "baseline.ttir").read_text()
        anchor = (OP / "round2/ir" / group / "optimized.ttir").read_text()
        parent = (
            anchor
            if previous is None
            else (previous / "ir" / group / "optimized.ttir").read_text()
        )
        folder = rd / "ir" / group
        for name, content in (
            ("baseline", baseline),
            ("anchor", anchor),
            ("parent", parent),
        ):
            immutable(folder / f"{name}.ttir", content)
        if round_name == "r3":
            if group.startswith("d128"):
                structure = remat(outer_unroll(parent))
            else:
                structure = dp_early(parent)
            variants = {"r3_structure": structure, "r3_query_pair": query_pair(parent)}
        elif round_name == "r4":
            variants = {
                "r4_qdo_cg": qdo_policy(parent, "cacheModifier = cg"),
                "r4_delta_early": move_delta(parent),
            }
        else:
            variants = {
                "r5_store_cs": re.sub(
                    r"(tt.store [^\n]+?) :", r"\1 cacheModifier = cs :", parent
                ),
                "r5_qdo_keep": qdo_policy(
                    parent, "cacheModifier = ca evictionPolicy = evict_last"
                ),
            }
        manifest[group] = {"candidates": {}}
        for name, text in variants.items():
            immutable(folder / f"{name}.ttir", text)
            manifest[group]["candidates"][name] = (
                __import__("hashlib").sha256(text.encode()).hexdigest()
            )
    immutable(rd / "manifest.json", json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("round", choices=["r3", "r4", "r5"])
    main(parser.parse_args().round)
