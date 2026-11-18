"""Count exact TTIR contents produced by official per-shape MM autotuning."""

import argparse
import json
import math
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import torch
import triton
from common import (
    ATOL,
    PERF_SHAPES,
    ROOT,
    RTOL,
    allocate,
    digest,
    frozen,
    gate,
    grid,
    runtime_args,
    save_json,
    sha256,
)
from environment import verify_distribution_files
from export_baseline import verify_extraction
from tutorial import get_cuda_autotune_config, matmul_kernel
from verify_pipeline import snapshot


def store_ir(directory, ck):
    text = ck.asm["ttir"]
    identifier = digest(text)
    path = directory / "ttir" / f"{identifier}.ttir"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        assert path.read_text() == text
    else:
        with path.open("x") as f:
            f.write(text)
    return identifier


def idle_except_self():
    # Called between shapes. This checks occupants without adding work inside
    # the autotuner's timing callback; it is not a device reservation.
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            os.environ["CUDA_VISIBLE_DEVICES"],
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        text=True,
    )
    others = [
        line for line in raw.splitlines() if int(line.split(",", 1)[0]) != os.getpid()
    ]
    assert not others, f"Foreign compute processes on selected GPU: {others}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    target = gate()
    frozen()
    verify_extraction()
    environment = verify_distribution_files()
    assert not triton.knobs.autotuning.cache, "Disable autotuning disk-result cache"
    idle_except_self()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    configs = get_cuda_autotune_config()
    tuner = triton.autotune(configs=configs, key=["M", "N", "K"], cache_results=False)(
        matmul_kernel
    )
    assert not tuner.cache_results
    save_json(
        out / "environment.json",
        dict(
            environment=environment,
            target=str(target),
            torch=torch.__version__,
            device_uuid=str(torch.cuda.get_device_properties(0).uuid),
            gpu_before=snapshot(),
            command=subprocess.list2cmdline([os.sys.executable, *os.sys.argv]),
            script_sha256=sha256(__file__),
            source_sha256=sha256(ROOT / "tutorial.py"),
            upstream_sha256=sha256(ROOT / "upstream/03-matrix-multiplication.py.txt"),
            frozen_baseline_sha256=sha256(ROOT / "baseline.json"),
            git_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            git_status=subprocess.check_output(
                ["git", "status", "--short"], cwd=ROOT, text=True
            ),
            options="Official 16 configs; key=M,N,K; default autotuner timer, no custom do_bench",
            scope="Separate counting experiment; does not replace baseline or claim optimization",
        ),
    )
    records = []
    for shape in PERF_SHAPES:
        idle_except_self()
        tensors = allocate(*shape)
        tensors[3].fill_(float("nan"))
        ck = tuner[lambda meta: grid(tensors, meta)](
            *runtime_args(tensors), ACTIVATION=""
        )
        selected = tuner.best_config
        identifier = store_ir(out, ck)
        reference = torch.mm(tensors[0], tensors[1])
        torch.testing.assert_close(tensors[2], reference, atol=ATOL, rtol=RTOL)
        assert torch.isfinite(tensors[2]).all()
        row = dict(
            shape=shape,
            selected_config=selected.all_kwargs(),
            selected_config_text=str(selected),
            selected_ttir_sha256=identifier,
            selected_cubin_sha256=digest(ck.asm["cubin"]),
            metadata=ck.metadata._asdict(),
            n_regs=ck.n_regs,
            n_spills=ck.n_spills,
            output_max_abs_error=(tensors[2].float() - reference.float())
            .abs()
            .max()
            .item(),
            correctness_passed=True,
            autotune_seconds=tuner.bench_time,
            config_timings_ms={
                str(c): [float(t) if math.isfinite(t) else None for t in times]
                for c, times in tuner.configs_timings.items()
            },
            infinite_timing_meaning="null denotes autotuner rejected config (e.g. resource/compile failure)",
        )
        idle_except_self()
        save_json(out / f"shape_{shape[0]}.json", row)
        records.append(row)
        print(
            f"{len(records)}/31 N={shape[0]} TTIR={identifier[:12]} {selected}",
            flush=True,
        )

    # Inspect every tried configuration's compiled output at every input
    # signature, including configurations rejected at the native loading step.
    # warmup compiles/loads cache entries without launching a GPU kernel.
    inventory = []
    for shape in PERF_SHAPES:
        idle_except_self()
        tensors = allocate(*shape)
        for config in configs:
            row = dict(shape=shape, config=config.all_kwargs(), config_text=str(config))
            try:
                ck = matmul_kernel.warmup(
                    *runtime_args(tensors),
                    ACTIVATION="",
                    **config.all_kwargs(),
                    grid=grid(tensors, config.kwargs),
                )
                row.update(
                    ttir_sha256=store_ir(out, ck), cubin_sha256=digest(ck.asm["cubin"])
                )
                ck._init_handles()
                row["status"] = "compiled_and_loadable"
            except Exception as exc:
                row.update(
                    status="failed", exception=type(exc).__name__, message=str(exc)
                )
            inventory.append(row)
    save_json(out / "configuration_inventory.json", inventory)
    groups = defaultdict(list)
    for row in records:
        groups[row["selected_ttir_sha256"]].append(row["shape"][0])
    result = dict(
        passed=True,
        shape_count=len(records),
        official_config_count=len(configs),
        distinct_selected_configs=len({row["selected_config_text"] for row in records}),
        distinct_selected_ttir=len(groups),
        selected_groups=[
            dict(ttir_sha256=h, square_sizes=ns) for h, ns in groups.items()
        ],
        distinct_all_compiled_ttir=len(
            {row["ttir_sha256"] for row in inventory if "ttir_sha256" in row}
        ),
        distinct_loadable_ttir=len(
            {
                row["ttir_sha256"]
                for row in inventory
                if row["status"] == "compiled_and_loadable"
            }
        ),
        inventory_failed_attempts=sum(row["status"] == "failed" for row in inventory),
        records=records,
        gpu_after=snapshot(),
        caveat="Exact TTIR text hashes including locations; JIT compilation caches allowed. "
        "Counts distinct contents, not fresh compiler executions; autotune winners can vary across runs.",
    )
    frozen()
    save_json(out / "summary.json", result)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ["records", "gpu_after"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
