"""Three real compiled routes measured in randomized same-round order."""

import argparse
import json
import random
import statistics
from pathlib import Path

import triton.testing
from common_port import (
    MATRIX,
    ROUTES,
    allocate,
    check_source,
    compile_routes,
    digest,
    group,
    input_hashes,
    launch,
    save,
    snapshot,
)

RATIOS = {
    "baseline_over_ir": ("baseline", "ir"),
    "baseline_over_source": ("baseline", "source"),
    "ir_over_source": ("ir", "source"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--accuracy", required=True)
    p.add_argument("--pipeline", required=True)
    p.add_argument("--seed", type=int, default=20)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--warmup", type=int, default=100)
    a = p.parse_args()
    out = Path(a.output)
    assert not out.exists()
    out.mkdir(parents=True)
    accuracy = json.loads(Path(a.accuracy).read_text())
    pipeline = json.loads(Path(a.pipeline).read_text())
    assert accuracy["passed"] and len(accuracy["checks"]) == 84
    assert pipeline["passed"] and len(pipeline["points"]) == 12
    assert accuracy["source_manifest"] == pipeline["source_manifest"] == check_source()
    rng = random.Random(a.seed)
    result = dict(
        environment=snapshot(),
        source_manifest=check_source(),
        seed=a.seed,
        rounds=a.rounds,
        rep_ms=a.rep,
        equal_warmup_ms=a.warmup,
        accuracy_sha256=digest(Path(a.accuracy)),
        pipeline_sha256=digest(Path(a.pipeline)),
        source_route="Actual Python-JIT CompiledKernel; not exported TTIR recompile",
        points=[],
    )
    for shape in MATRIX:
        d, c = shape[-2:]
        g = group(d, c)
        t = allocate(*shape, seed=a.seed)
        kernels, metas = compile_routes(t)
        for route in ROUTES:
            old = accuracy["compile"][g][route]
            assert metas[route]["hashes"] == old["hashes"]
            assert metas[route]["elf_text_sections"] == old["elf_text_sections"]
        hashes = input_hashes(t)
        matching = [
            x
            for x in accuracy["checks"]
            if x["shape"] == list(shape) and x["distribution"] == "normal"
        ]
        assert len(matching) == 3 and all(x["input_sha256"] == hashes for x in matching)

        def measure(route, rep=a.rep):
            return triton.testing.do_bench_cudagraph(
                lambda: launch(kernels[route], t), rep=rep
            )

        for route in ROUTES:
            measure(route, a.warmup)
        aa = []
        for _ in range(3):
            first = measure("baseline")
            second = measure("baseline")
            aa.append(dict(first_ms=first, second_ms=second, ratio=first / second))
        samples = []
        for _ in range(a.rounds):
            order = list(ROUTES)
            rng.shuffle(order)
            times = {route: measure(route) for route in order}
            samples.append(
                dict(
                    order=order,
                    ms=times,
                    ratios={k: times[x] / times[y] for k, (x, y) in RATIOS.items()},
                )
            )
        item = dict(
            shape=shape,
            group=g,
            input_sha256=hashes,
            metadata=metas,
            aa=aa,
            samples=samples,
            median_us={
                r: 1000 * statistics.median(s["ms"][r] for s in samples) for r in ROUTES
            },
            median_ratios={
                k: statistics.median(s["ratios"][k] for s in samples) for k in RATIOS
            },
        )
        result["points"].append(item)
        save(out / "benchmark.json", result)
        print("bench", shape, item["median_us"], item["median_ratios"], flush=True)


if __name__ == "__main__":
    main()
