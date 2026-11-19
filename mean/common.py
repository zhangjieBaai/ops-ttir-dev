"""Frozen shape mapping, one native compiler, one launcher, and evidence helpers."""

import fcntl
import hashlib
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

os.environ["TRITON_DISABLE_LINE_INFO"] = "1"

import torch
import triton
from environment import check_environment
from triton.backends.nvidia.compiler import CUDABackend

ROOT = Path(__file__).resolve().parent
PERF_SHAPES = [
    (1, 16, 1),
    (64, 1008, 1),
    (1024, 4096, 1),
    (32, 8192, 1),
    (8, 32768, 1),
    (16, 65536, 1),
    (1, 16, 16),
    (16, 127, 64),
    (64, 512, 128),
    (8, 2048, 512),
    (16, 4096, 1023),
    (64, 257, 32),
    (1, 1, 1024),
    (1, 16, 1024),
    (1, 64, 1024),
    (1, 256, 1024),
    (1, 1024, 1024),
    (8, 64, 1024),
    (64, 256, 1024),
    (1, 63, 1025),
    (8, 257, 2048),
    (32, 512, 2048),
    (1, 1024, 4096),
    (8, 256, 4096),
    (16, 64, 8192),
    (1, 512, 8192),
    (4, 1024, 8192),
    (64, 16, 16384),
    (1, 256, 32768),
    (4, 64, 65536),
]
BOUNDARIES = [
    (3, n, k)
    for n, k in [
        (1, 1),
        (15, 1),
        (17, 1),
        (32769, 1),
        (1, 2),
        (63, 17),
        (65, 1023),
        (15, 1024),
        (17, 1025),
        (255, 2049),
        (257, 4095),
        (1025, 1024),
    ]
]
ALL_SHAPES = PERF_SHAPES + BOUNDARIES


def key(shape):
    return "x".join(map(str, shape))


def digest(value):
    return hashlib.sha256(
        value if isinstance(value, bytes) else value.encode()
    ).hexdigest()


def sha256(path):
    return digest(Path(path).read_bytes())


def normalized(value):
    return json.loads(json.dumps(value, default=str))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, default=str) + "\n")


def snapshot():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,pci.bus_id,clocks.sm,clocks.mem,"
            "temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used",
            "--format=csv",
        ],
        text=True,
    )


@lru_cache(maxsize=1)
def gate():
    check_environment()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert visible and "," not in visible, "Pin one idle device"
    assert not triton.knobs.compilation.override
    assert triton.knobs.runtime.add_stages_inspection_hook is None
    target = triton.runtime.driver.active.get_current_target()
    assert target.backend == "cuda" and target.arch == 90
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "-i", visible, "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    lock = open("/root/tmp/mean-" + gpu_uuid + ".lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gate.lock = lock
    gate.uuid = gpu_uuid
    return target


def check_contention():
    gate()
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader",
        ],
        text=True,
    )
    foreign = [
        line
        for line in apps.splitlines()
        if line.startswith(gate.uuid) and int(line.split(",")[1]) != os.getpid()
    ]
    assert not foreign, f"Other compute processes on selected GPU: {foreign}"
    return apps


def original_config(shape):
    import frozen_heuristics as h

    m, n, k = shape
    args = dict(M=m, N=n, K=k)
    if k >= 1024:
        return dict(BLOCK_SIZE_K=128, VEC_SIZE=8), dict(
            num_warps=8, num_ctas=1, num_stages=3, enable_fp_fusion=True
        )
    if k == 1:
        args["TILE_N"] = h.softmax_heur_tile_n_inner(args)
        warps = h.softmax_heur_num_warps_inner(args)
    else:
        args["TILE_K"] = h.mean_heur_tile_k(args)
        args["TILE_N"] = h.mean_heur_tile_n_non_inner(args)
        warps = h.softmax_heur_num_warps_non_inner(args)
    args["ONE_TILE_PER_CTA"] = h.mean_heur_one_tile_per_cta(args)
    meta = {
        k: v
        for k, v in args.items()
        if k.startswith("TILE_") or k == "ONE_TILE_PER_CTA"
    }
    return meta, dict(num_warps=warps, num_ctas=1, num_stages=3, enable_fp_fusion=True)


def allocate(shape, guarded=False):
    m, n, k = shape
    x = torch.randn((m, n, k), device="cuda", dtype=torch.float16)
    storage = torch.full(
        (m * k + (32 if guarded else 0),), 12345.0, device="cuda", dtype=x.dtype
    )
    y = (storage[16:-16] if guarded else storage).view(m, k)
    return x, y, storage


def launch_source(shape, tensors):
    import tutorial

    x, y = tensors[:2]
    m, n, k = shape
    meta, options = original_config(shape)
    fn = (
        tutorial.mean_dim_kernel_inner if k == 1 else tutorial.mean_dim_kernel_non_inner
    )
    if k >= 1024:
        fn = tutorial.mean_dim_kernel_non_inner_vec
    width = meta.get("TILE_K", meta.get("BLOCK_SIZE_K", 1) * meta.get("VEC_SIZE", 1))
    grid = (m, 1 if k == 1 else triton.cdiv(k, width), 1)
    return fn[grid](y, x, m, n, *([] if k == 1 else [k]), **meta, **options)


@lru_cache(maxsize=1)
def manifest():
    data = json.loads((ROOT / "baseline.json").read_text())
    for path, expected in data["frozen_files"].items():
        assert sha256(ROOT / path) == expected, f"Frozen input changed: {path}"
    return data


def group_for(shape):
    data = manifest()
    point = data["mapping"][key(shape)]
    return point["group"], data["groups"][point["group"]]


def ir_path(group, name):
    return ROOT / "ir" / group / f"{name}.ttir"


def kernel_info(ck, group, name):
    path = ROOT / "artifacts" / group / name / ck.hash
    path.mkdir(parents=True, exist_ok=True)
    for stage, value in ck.asm.items():
        raw = value if isinstance(value, bytes) else value.encode()
        dest = path / f"kernel.{stage}"
        if dest.exists():
            assert dest.read_bytes() == raw
        else:
            dest.write_bytes(raw)
    return dict(
        hash=ck.hash,
        stages={k: digest(v) for k, v in ck.asm.items()},
        metadata=ck.metadata._asdict(),
        n_regs=ck.n_regs,
        n_spills=ck.n_spills,
        input_ttir=str(ir_path(group, name)),
        input_sha256=sha256(ir_path(group, name)),
    )


@lru_cache(maxsize=None)
def compile_ir(group, name="baseline"):
    target = gate()
    config = manifest()["groups"][group]
    path = ir_path(group, name)
    options = CUDABackend(target).parse_options(config["options"])
    assert options.ir_override is None
    assert normalized(vars(options)) == config["parsed_options"]
    ck = triton.compile(str(path), options=config["options"])
    assert ck.src.ext == "ttir" and ck.src.src == path.read_text()
    assert ck.metadata.target == target
    for k, v in vars(options).items():
        assert normalized(ck.metadata._asdict()[k]) == normalized(v), k
    ck._init_handles()
    kernel_info(ck, group, name)
    return ck


def launch(ck, shape, tensors):
    group, config = group_for(shape)
    x, y = tensors[:2]
    m, n, k = shape
    assert x.shape == (m, n, k) and y.shape == (m, k)
    assert x.is_contiguous() and y.is_contiguous()
    assert x.is_cuda and y.device == x.device and x.dtype == y.dtype == torch.float16
    assert x.data_ptr() % 16 == y.data_ptr() % 16 == 0
    assert x.untyped_storage().data_ptr() != y.untyped_storage().data_ptr()
    assert 0 < m * n * k < 2**31
    assert ck.metadata.num_warps == config["options"]["num_warps"]
    args = [y, x, m, n] + ([] if k == 1 else [k])
    ck[tuple(manifest()["mapping"][key(shape)]["grid"])](
        *[args[i] for i in config["runtime_indices"]]
    )


def receipt():
    return dict(
        command=[sys.executable, *sys.argv],
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        git_status=subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        protocol_sha256=sha256(ROOT / "optimization_plan.md"),
        baseline_sha256=sha256(ROOT / "baseline.json"),
        code_sha256={p.name: sha256(p) for p in ROOT.glob("*.py")},
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        physical_uuid=gate.uuid,
        gpu_snapshot=snapshot(),
    )
