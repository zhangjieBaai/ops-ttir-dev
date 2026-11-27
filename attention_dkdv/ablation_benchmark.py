"""Full32 paired ablation with add/remove controls for the frozen r6 final."""

import argparse
import os
import random
import statistics
import sys
from pathlib import Path

import torch
import triton
import triton.testing

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "round2"))
sys.path.insert(0, str(ROOT / "multiround"))
from common_round2 import (  # noqa: E402
    MATRIX,
    OPTIONS,
    _attn_bwd_dkdv_only,
    allocate,
    digest,
    env,
    group,
    input_hashes,
    launch,
    launch_source,
    save,
    snapshot,
    text_sections,
)
from make_ablation import variants  # noqa: E402
from warmup import GraphWarmup  # noqa: E402


def check(kernels, tensors):
    source = launch_source(_attn_bwd_dkdv_only, tensors)
    expected = [tensors[k].clone() for k in ("dk", "dv")]
    before = input_hashes(tensors)
    for key, value in zip(("dk", "dv"), expected):
        assert torch.isfinite(value).all(), key
        torch.testing.assert_close(
            value.double(),
            tensors["reference"][int(key == "dv")],
            rtol=0.03,
            atol=0.008,
        )
    for name, kernel in kernels.items():
        for key in ("dk", "dv"):
            tensors[key].fill_(float("nan"))
        launch(kernel, tensors)
        assert all(
            torch.equal(tensors[k].view(torch.int16), value.view(torch.int16))
            for k, value in zip(("dk", "dv"), expected)
        ), name
        assert input_hashes(tensors) == before, name
    return source, before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    env.check_environment()
    import json

    pipeline = json.loads(args.pipeline.read_text())
    assert pipeline["passed"] and len(pipeline["points"]) == 32
    assert os.environ.get("ATTENTION_R2_LOCK_SM_MHZ") == "1200"
    environment = snapshot()
    compiled, resources = {}, {}
    for d in (64, 128):
        for causal in (False, True):
            g = group(d, causal)
            compiled[g], resources[g] = {}, {}
            for name, path in variants(d, causal).items():
                kernel = triton.compile(str(path), options=OPTIONS)
                kernel._init_handles()
                compiled[g][name] = kernel
                resources[g][name] = dict(
                    path=str(path.relative_to(ROOT)),
                    ttir_sha256=digest(path),
                    hashes={k: digest(v) for k, v in kernel.asm.items()},
                    elf_text=text_sections(kernel.asm["cubin"]),
                    registers=kernel.n_regs,
                    spills=kernel.n_spills,
                    shared_bytes=kernel.metadata.shared,
                )
    accuracy = dict(
        environment=environment, pipeline_sha256=digest(args.pipeline), cases=[]
    )
    # Complete numerical gate for every normal point and all group stress cases before timing.
    cases = [(shape, "normal") for shape in MATRIX]
    cases += [
        ((1, 4, 128, d, c), distribution)
        for d in (64, 128)
        for c in (False, True)
        for distribution in ("zero_do", "constant", "large")
    ]
    for shape, distribution in cases:
        g = group(shape[3], shape[4])
        tensors = allocate(*shape, distribution=distribution, seed=args.seed)
        source, hashes = check(compiled[g], tensors)
        assert all(
            source.asm[k] == compiled[g]["baseline"].asm[k]
            for k in ("llir", "ptx", "cubin")
        )
        accuracy["cases"].append(
            dict(
                shape=shape,
                distribution=distribution,
                names=list(compiled[g]),
                outputs_bitwise_equal=True,
                input_hashes=hashes,
                math_reference_passed=True,
            )
        )
        print("ACCURACY", shape, distribution, flush=True)
    accuracy["passed"] = True
    accuracy["kernel_checks"] = sum(len(c["names"]) for c in accuracy["cases"])
    save(args.output / "accuracy.json", accuracy)
    save(args.output / "resources.json", resources)
    result = dict(
        environment=environment,
        seed=args.seed,
        rounds=args.rounds,
        rep_ms=20,
        pipeline_sha256=digest(args.pipeline),
        accuracy_sha256=digest(args.output / "accuracy.json"),
        resources_sha256=digest(args.output / "resources.json"),
        points=[],
    )
    rng = random.Random(args.seed)
    for shape in MATRIX:
        g = group(shape[3], shape[4])
        tensors = allocate(*shape, seed=args.seed)
        kernels = compiled[g]
        warmers = {
            name: GraphWarmup(lambda kernel=kernel: launch(kernel, tensors))
            for name, kernel in kernels.items()
        }
        initial = {name: warmer.run(500) for name, warmer in warmers.items()}
        samples = []
        for _ in range(args.rounds):
            order = list(kernels) + ["baseline_copy"]
            rng.shuffle(order)
            times, warmups = {}, {}
            for label in order:
                name = "baseline" if label == "baseline_copy" else label
                warmups[label] = warmers[name].run(100)
                times[label] = 1000 * triton.testing.do_bench_cudagraph(
                    lambda: launch(kernels[name], tensors), rep=20
                )
            samples.append(dict(order=order, us=times, warmups=warmups))
        aa = statistics.median(
            s["us"]["baseline"] / s["us"]["baseline_copy"] for s in samples
        )
        result["points"].append(
            dict(
                shape=shape,
                group=g,
                samples=samples,
                initial_warmup=initial,
                aa=aa,
                ttir_sha256={k: v["ttir_sha256"] for k, v in resources[g].items()},
            )
        )
        save(args.output / "benchmark.json", result)
        assert 0.97 <= aa <= 1.03, (shape, aa)
        print("PERF", shape, aa, flush=True)
    result["passed"] = True
    save(args.output / "benchmark.json", result)


if __name__ == "__main__":
    main()
