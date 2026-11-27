"""Three-way source/TTIR comparison using the original frozen launch contract."""

import argparse
import json
import random
import statistics
import struct
import sys
from pathlib import Path

import membership_source_v1 as optimized_source
import torch
import triton.language as tl
from triton.backends.nvidia.compiler import CUDABackend
from triton.testing import do_bench_cudagraph

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import common as c  # noqa: E402

NAMES = ("baseline", "ir", "source")
RATIOS = {
    "baseline_over_ir": ("baseline", "ir"),
    "baseline_over_source": ("baseline", "source"),
    "ir_over_source": ("ir", "source"),
}


def tensor_hash(x):
    return c.sha(x.detach().cpu().contiguous().numpy().tobytes())


def elf_text(data):
    assert data[:6] == b"\x7fELF\x02\x01", "Expected little-endian ELF64"
    offset = struct.unpack_from("<Q", data, 40)[0]
    size, count, strings_index = struct.unpack_from("<HHH", data, 58)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", data, offset + i * size) for i in range(count)
    ]
    section = sections[strings_index]
    names = data[section[4] : section[4] + section[5]]
    result = {}
    for item in sections:
        end = names.index(b"\0", item[0])
        name = names[item[0] : end].decode()
        if name.startswith(".text"):
            result[name] = c.sha(data[item[4] : item[4] + item[5]])
    assert result
    return result


def complete_options(kernel):
    backend = CUDABackend(kernel.metadata.target)
    metadata = kernel.metadata._asdict()
    return {name: metadata[name] for name in vars(backend.parse_options({}))}


def source_stage2(t, warps):
    m, e = t["x"].shape
    _, args = c.runtime_args(t)
    return optimized_source.group_idx_and_topk_triton[(m,)](
        *args[:12],
        1,
        *args[12:],
        N_GROUP=8,
        TOPK_GROUP=4,
        TOPK=8,
        BLOCK_GROUP=8,
        BLOCK_EXPERT=e,
        INPUT_DTYPE=tl.float32,
        renormalize=1,
        **c.options(warps),
    )


def launch(name, kernels, t):
    if name != "source":
        return c.launch(kernels, t)
    # Native source launchers retain the original argument positions, including
    # constexpr slots. The compiled device ABI and grids remain identical.
    m, e = t["x"].shape
    a, b = c.runtime_args(t)
    kernels[0][(m * 8, 1, 1)](*a, 1, e // 8, tl.float32)
    kernels[1][(m, 1, 1)](*b[:12], 1, *b[12:], 8, 4, 8, 8, e, tl.float32, 1)


def artifact(kernel, path):
    path.mkdir(parents=True, exist_ok=True)
    kernel._init_handles()
    for ext, data in kernel.asm.items():
        out = path / f"kernel.{ext}"
        out.write_bytes(data) if isinstance(data, bytes) else out.write_text(data)
    return dict(
        hashes={ext: c.sha(data) for ext, data in kernel.asm.items()},
        elf_text=elf_text(kernel.asm["cubin"]),
        metadata=kernel.metadata._asdict(),
        complete_options=complete_options(kernel),
        regs=kernel.n_regs,
        spills=kernel.n_spills,
    )


def freeze_checks():
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    checks = {
        "source_sha256": ROOT / manifest["source_file"],
        "parent_source_sha256": ROOT.parent / "frozen_kernels.py",
        "parent_manifest_sha256": ROOT.parent / "baseline.json",
        "parent_selection_sha256": ROOT.parent / "selection.json",
        "protocol_sha256": ROOT / "optimization_plan.md",
    }
    for name, path in checks.items():
        assert c.sha(path) == manifest[name], (name, path)
    return manifest


def prepare(item, seed):
    m, e, w = (item[name] for name in ("tokens", "experts", "warps"))
    t = c.allocate(m, e, seed=seed)
    original = c.source_launch(t, w)
    original_outputs = {name: t[name].clone() for name in ("g", "w", "i")}
    baseline = c.compile_pair(e, w, "baseline")
    ir = c.compile_pair(e, w, "membership")
    c.launch(baseline, t)
    torch.cuda.synchronize()
    assert all(
        torch.equal(original_outputs[name], t[name]) for name in original_outputs
    )
    gates = []
    for stage, (src, file) in enumerate(zip(original, baseline), 1):
        path = ROOT.parent / "ir" / c.group(e, w) / f"baseline_s{stage}.ttir"
        assert src.asm["ttir"] == path.read_text()
        assert all(src.asm[ext] == file.asm[ext] for ext in ("llir", "ptx", "cubin"))
        assert complete_options(src) == complete_options(file)
        gates.append(
            dict(stage=stage, source_file_stages_identical=True, options_identical=True)
        )
    # This is the actual JIT-produced object used by the source timing path.
    source_kernel = source_stage2(t, w)
    source = (original[0], source_kernel)
    frozen = ROOT / "ir" / c.group(e, w) / "source_s2.ttir"
    c.immutable(frozen, source_kernel.asm["ttir"])
    recompiled = c.triton.compile(str(frozen), options=c.options(w))
    recompiled._init_handles()
    assert all(
        source_kernel.asm[ext] == recompiled.asm[ext]
        for ext in ("llir", "ptx", "cubin")
    )
    assert complete_options(source_kernel) == complete_options(recompiled)
    assert all(
        complete_options(baseline[1]) == complete_options(k)
        for k in (ir[1], source_kernel)
    )
    assert all(
        baseline[1].metadata.target == k.metadata.target for k in (ir[1], source_kernel)
    )
    kernels = dict(baseline=baseline, ir=ir, source=source)
    artifacts = {
        name: [
            artifact(kernel, ROOT / "artifacts" / c.group(e, w) / name / f"s{stage}")
            for stage, kernel in enumerate(pair, 1)
        ]
        for name, pair in kernels.items()
    }
    ttir = source_kernel.asm["ttir"]
    pipeline = dict(
        original_baseline=gates,
        original_outputs_identical=True,
        optimized_source_file_stages_identical=True,
        optimized_source_file_options_identical=True,
        all_three_options_identical=True,
        source_timing_object="original stage1 JIT CompiledKernel + optimized stage2 JIT CompiledKernel",
        exported_source_file_used_for_timing=False,
        stage2_ir_source={
            "same_elf_text": artifacts["ir"][1]["elf_text"]
            == artifacts["source"][1]["elf_text"],
            "same_stages": {
                ext: ir[1].asm[ext] == source_kernel.asm[ext]
                for ext in ("ttir", "ttgir", "llir", "ptx", "cubin")
            },
            "source_has_broadcast_shape": f"tensor<8x{e // 8}xi1>" in ttir,
            "source_has_dense_membership_shape": f"tensor<{e}x8xi32>" in ttir,
        },
    )
    return kernels, dict(pipeline=pipeline, artifacts=artifacts)


def validate(kernels, item, seed):
    m, e = item["tokens"], item["experts"]
    records = []
    for dist in (
        "normal",
        "ties",
        "group_ties",
        "bias_dominant",
        "saturated",
        "negative",
    ):
        t = c.allocate(m, e, dist, seed)
        inputs = {name: tensor_hash(t[name]) for name in ("x", "b")}
        ref_w, ref_i = c.reference(t)
        c.launch(kernels["baseline"], t)
        base = {name: t[name].clone() for name in ("g", "w", "i")}
        for name, pair in kernels.items():
            for output in ("g", "w"):
                t[output].fill_(float("nan"))
            t["i"].fill_(-777)
            launch(name, pair, t)
            torch.cuda.synchronize()
            c.check_guards(t)
            assert inputs == {key: tensor_hash(t[key]) for key in ("x", "b")}
            assert all(
                torch.equal(t[key].view(torch.int32), base[key].view(torch.int32))
                for key in base
            )
            assert torch.equal(t["i"].cpu().long(), ref_i), (m, e, dist, name)
            torch.testing.assert_close(t["w"].cpu(), ref_w, rtol=2e-5, atol=2e-6)
            records.append(
                dict(
                    variant=name,
                    shape=[m, e],
                    distribution=dist,
                    seed=seed,
                    input_hashes=inputs,
                    outputs_bitwise_baseline=True,
                    indices_exact_cpu=True,
                    max_weight_abs_error_cpu=(t["w"].cpu() - ref_w).abs().max().item(),
                    guards_and_input_unchanged=True,
                    status="PASS",
                )
            )
    return records


def measure(kernels, t, rounds, rep, seed):
    rng = random.Random(seed)
    for name, pair in kernels.items():
        for _ in range(10):
            launch(name, pair, t)
    torch.cuda.synchronize()
    do_bench_cudagraph(lambda: c.launch(kernels["baseline"], t), rep=50)
    samples = []
    aa = []
    for rnd in range(rounds):
        aa_values = [
            do_bench_cudagraph(lambda: c.launch(kernels["baseline"], t), rep=rep) * 1000
            for _ in range(2)
        ]
        aa.append(dict(round=rnd, us=aa_values, ratio=aa_values[0] / aa_values[1]))
        order = list(NAMES)
        rng.shuffle(order)
        values = {
            name: do_bench_cudagraph(lambda: launch(name, kernels[name], t), rep=rep)
            * 1000
            for name in order
        }
        ratios = {name: values[a] / values[b] for name, (a, b) in RATIOS.items()}
        samples.append(dict(round=rnd, order=order, us=values, ratios=ratios))
    return dict(
        samples=samples,
        aa=aa,
        median_ratios={
            name: statistics.median(s["ratios"][name] for s in samples)
            for name in RATIOS
        },
        median_us={
            name: statistics.median(s["us"][name] for s in samples) for name in NAMES
        },
        aa_median=statistics.median(row["ratio"] for row in aa),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--seed", type=int, default=202611)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    folder = ROOT / "results" / args.run
    assert not folder.exists(), folder
    folder.mkdir()
    result = dict(
        environment=c.environment(),
        source_manifest=freeze_checks(),
        harness_sha256=c.sha(Path(__file__)),
        seed=args.seed,
        rounds=args.rounds,
        rep_ms=args.rep,
        check_only=args.check_only,
        records=[],
    )
    mapping = json.loads((ROOT.parent / "baseline.json").read_text())["mapping"]
    for index, item in enumerate(mapping.values()):
        seed = args.seed + 1009 * index
        kernels, evidence = prepare(item, seed)
        accuracy = validate(kernels, item, seed + 1)
        record = dict(item, seed=seed, evidence=evidence, accuracy=accuracy)
        if not args.check_only:
            t = c.allocate(item["tokens"], item["experts"], seed=seed)
            record["benchmark_input_hashes"] = {
                name: tensor_hash(t[name]) for name in ("x", "b")
            }
            record["timing"] = measure(kernels, t, args.rounds, args.rep, seed)
            print(
                item["tokens"],
                item["experts"],
                record["timing"]["median_ratios"],
                flush=True,
            )
        result["records"].append(record)
        c.save(folder / "results.json", result)
    result["passed"] = True
    if not args.check_only:
        result["geomean_ratios"] = {
            name: statistics.geometric_mean(
                row["timing"]["median_ratios"][name] for row in result["records"]
            )
            for name in RATIOS
        }
        result["aa_passed"] = all(
            0.97 <= row["timing"]["aa_median"] <= 1.03 for row in result["records"]
        )
    c.save(folder / "results.json", result)
    print(result.get("geomean_ratios", "accuracy PASS"), flush=True)


if __name__ == "__main__":
    main()
