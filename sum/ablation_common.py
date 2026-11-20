"""Frozen ablation routes and native-TTIR/public-source controls."""

import json
import struct

import triton
from common import (
    OPTIONS,
    PERF_SIZES,
    ROOT,
    frozen,
    geometry,
    kernel_info,
    save_json,
    sha256,
    write_once,
)
from round2 import compile_path, isolated

BASE = ROOT / "ablation"
DIRECT = [
    "baseline",
    "i32",
    "fold",
    "i32_fold",
    "cg",
    "cs",
    "tail",
    "wide_two",
    "single",
    "single_tail",
]
TIMED = [*DIRECT, "py_single"]
PUBLIC = [name for name in DIRECT if name != "cs"]
VALIDATED = [*DIRECT, *["py_" + name for name in PUBLIC]]
EDGES = [("baseline", name) for name in TIMED if name != "baseline"] + [
    ("i32", "i32_fold"),
    ("fold", "i32_fold"),
    ("wide_two", "single"),
    ("single", "single_tail"),
    ("single", "py_single"),
]


def signature_indices(ck):
    return [index for index in range(3) if (index,) not in ck.src.constants]


def text_sections(binary):
    """Read actual ELF64 .text bytes; debug sections are deliberately separate."""
    assert binary[:6] == b"\x7fELF\x02\x01"
    offset = struct.unpack_from("<Q", binary, 40)[0]
    entry_size, count, names_index = struct.unpack_from("<HHH", binary, 58)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", binary, offset + i * entry_size)
        for i in range(count)
    ]
    names_section = sections[names_index]
    names = binary[names_section[4] : names_section[4] + names_section[5]]
    result = {}
    from common import digest

    for section in sections:
        name = names[section[0] :].split(b"\0", 1)[0].decode()
        if name.startswith(".text."):
            result[name] = digest(binary[section[4] : section[4] + section[5]])
    assert result
    return result


def spec(path, kind, n, runtime_indices):
    return dict(
        path=str(path.relative_to(ROOT)),
        sha256=sha256(path),
        kind=kind,
        grid=[geometry(n)[1], 1, 1] if kind == "stage1" else [1, 1, 1],
        runtime_indices=runtime_indices,
    )


def direct_route(n, name, original, single_mapping):
    groups = original["shapes"][str(n)]
    if name in ["single", "single_tail", "wide_two"] and n <= 16384:
        single = single_mapping["routes"][str(n)]
        path = ROOT / single["path"]
        record = spec(
            path,
            "wide_mid" if name == "wide_two" else "single",
            n,
            single["runtime_indices"],
        )
        if name == "wide_two":
            copy_group = original["shapes"]["1"][1]
            copy = original["groups"][copy_group]
            return [
                record,
                spec(ROOT / copy["path"], "copy", n, copy["runtime_indices"]),
            ]
        return [record]
    mapped = {
        "i32": "r1_i32",
        "fold": "r1_fold",
        "i32_fold": "r1_i32_fold",
        "cg": "r1_cg",
        "cs": "r1_cs",
        "tail": "r2_tail",
        "single_tail": "r2_tail",
    }
    records = []
    for stage, group in enumerate(groups):
        config = original["groups"][group]
        path = ROOT / config["path"]
        if name in mapped:
            path = path.with_name(mapped[name] + ".ttir")
        records.append(
            spec(
                path, "stage1" if stage == 0 else "stage2", n, config["runtime_indices"]
            )
        )
    return records


def arguments(record, tensors):
    x, mid, out = tensors[:3]
    kind = record["kind"]
    if kind == "single":
        return x, out, x.numel()
    if kind in ["stage1", "wide_mid"]:
        return x, mid, x.numel()
    return mid, out, 1 if kind == "copy" else mid.numel()


def launch(kernels, records, tensors):
    for record in records:
        values = arguments(record, tensors)
        kernels[record["path"]][tuple(record["grid"])](
            *[values[i] for i in record["runtime_indices"]]
        )


def source_stage(record, tensors, name):
    import ablation_triton as public

    values = arguments(record, tensors)
    n = tensors[0].numel()
    block, _, tail = geometry(n)
    first = record["kind"] in ["stage1", "wide_mid", "single"]
    if first:
        width = (
            triton.next_power_of_2(n)
            if record["kind"] in ["wide_mid", "single"]
            else block
        )
        ck = public.sum_kernel_1[tuple(record["grid"])](
            *values,
            width,
            I32=name in ["i32", "i32_fold"],
            FOLD=name in ["fold", "i32_fold"],
            CACHE="." + name if name in ["cg", "cs"] else "",
            TAIL=name in ["tail", "single_tail"] and record["kind"] == "stage1",
            **OPTIONS,
        )
    else:
        ck = public.sum_kernel_2[tuple(record["grid"])](
            *values,
            1 if record["kind"] == "copy" else tail,
            FOLD=name in ["fold", "i32_fold"],
            CACHE="." + name if name in ["cg", "cs"] else "",
            **OPTIONS,
        )
    return ck


def export_public():
    from common import allocate

    original = frozen()
    single_mapping = json.loads((ROOT / "round2_manifest.json").read_text())
    assert not (BASE / "manifest.json").exists()
    direct, python, checks, failures = {}, {}, [], []
    for name in DIRECT:
        direct[name] = {}
        if name in PUBLIC:
            python["py_" + name] = {}
        for n in PERF_SIZES:
            records = direct_route(n, name, original, single_mapping)
            direct[name][str(n)] = records
            tensors = allocate(n)
            if name == "cs":
                if n == 16:
                    try:
                        source_stage(records[0], tensors, name)
                    except Exception as error:
                        failures.append(
                            dict(
                                name=name,
                                n=n,
                                status="PUBLIC_LOAD_CACHE_UNSUPPORTED",
                                error=str(error),
                            )
                        )
                    else:
                        raise AssertionError(
                            "Unexpected .cs public load support; reassess protocol"
                        )
                continue
            public_records = []
            for stage, record in enumerate(records):
                source = source_stage(record, tensors, name)
                from common import digest

                path = BASE / "ir" / (digest(source.asm["ttir"]) + ".ttir")
                write_once(path, source.asm["ttir"])
                exported = dict(
                    record,
                    path=str(path.relative_to(ROOT)),
                    sha256=sha256(path),
                    runtime_indices=signature_indices(source),
                )
                public_records.append(exported)
                compiled = compile_path(exported)
                same = {
                    ext: source.asm[ext] == compiled.asm[ext]
                    for ext in ["llir", "ptx", "cubin"]
                }
                assert all(same.values()), (name, n, stage, same)
                direct_kernel = compile_path(record)
                left, right = text_sections(direct_kernel.asm["cubin"]), text_sections(
                    compiled.asm["cubin"]
                )
                checks.append(
                    dict(
                        name=name,
                        n=n,
                        stage=stage + 1,
                        file_pipeline_identical=same,
                        direct_text=left,
                        python_text=right,
                        executable_text_identical=left == right,
                        direct=kernel_info(direct_kernel, ROOT / record["path"]),
                        python=kernel_info(compiled, path),
                    )
                )
            python["py_" + name][str(n)] = public_records
            print(f"Export {name} n={n}", flush=True)
    data = dict(
        experiment="sum_ablation_v1",
        plan_sha256=sha256(ROOT / "ablation_plan.md"),
        source_sha256=sha256(ROOT / "ablation_triton.py"),
        baseline_manifest_sha256=sha256(ROOT / "baseline.json"),
        round2_manifest_sha256=sha256(ROOT / "round2_manifest.json"),
        routes={**direct, **python},
        checks=checks,
        public_failures=failures,
    )
    save_json(BASE / "manifest.json", data)


def mapping():
    data = json.loads((BASE / "manifest.json").read_text())
    assert data["plan_sha256"] == sha256(ROOT / "ablation_plan.md")
    assert data["source_sha256"] == sha256(ROOT / "ablation_triton.py")
    assert data["baseline_manifest_sha256"] == sha256(ROOT / "baseline.json")
    assert data["round2_manifest_sha256"] == sha256(ROOT / "round2_manifest.json")
    return data


def load(names, data):
    kernels = {}
    for name in names:
        for records in data["routes"][name].values():
            for record in records:
                if record["path"] not in kernels:
                    kernels[record["path"]] = compile_path(record)
    return kernels


def identities(names, data, kernels):
    return {
        name: {
            n: [kernel_info(kernels[r["path"]], ROOT / r["path"]) for r in records]
            for n, records in data["routes"][name].items()
        }
        for name in names
    }


__all__ = ["isolated"]
