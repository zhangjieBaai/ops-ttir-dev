"""Shared native TTIR compiler, row launcher and test allocation helpers."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["TRITON_DISABLE_LINE_INFO"] = "1"

import torch  # noqa: E402
import triton  # noqa: E402

ROOT = Path(__file__).resolve().parent
CONTRACT = json.loads((ROOT / "contract.json").read_text())
OPTIONS = CONTRACT["options"]
SHAPES = CONTRACT["performance"]["shapes"]
spec = importlib.util.spec_from_file_location(
    "ttir_environment", ROOT.parent / "layer_norm/environment.py"
)
environment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(environment)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_data(data):
    return hashlib.sha256(
        data if isinstance(data, bytes) else data.encode()
    ).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), f"Immutable evidence already exists: {path}"
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def telemetry():
    fields = (
        "index,uuid,name,driver_version,pstate,clocks.sm,clocks.mem,"
        "power.draw,temperature.gpu,utilization.gpu,memory.used"
    )
    return subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv"], text=True
    )


def provenance():
    return dict(
        command=[sys.executable, *sys.argv],
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        contract_sha256=digest(ROOT / "contract.json"),
        scripts={p.name: digest(p) for p in ROOT.glob("*.py")},
        environment={
            k: v
            for k, v in os.environ.items()
            if any(
                s in k
                for s in ["CUDA", "TRITON", "PYTHON", "LD_LIBRARY", "FLAG", "PATH"]
            )
        },
        telemetry=telemetry(),
        torch_version=torch.__version__,
        torch_path=torch.__file__,
        loaded_libraries=sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if any(
                    s in line
                    for s in ["libcuda", "libtriton", "libnv", "libtorch_cuda"]
                )
            }
        ),
        clocks_locked="NOT_VERIFIED; no clock-setting command issued",
        cache="no explicit cache flush; repeated same-input CUDA graph replay",
    )


def compile_ir(name):
    environment.check_environment()
    assert not triton.knobs.compilation.override
    assert triton.knobs.runtime.add_stages_inspection_hook is None
    target = triton.runtime.driver.active.get_current_target()
    assert target.backend == "cuda" and target.arch == 90
    path = ROOT / f"ir/{name}.ttir"
    ck = triton.compile(str(path), options=OPTIONS)
    assert not ck.metadata.ir_override
    ck._init_handles()
    folder = ROOT / f"artifacts/{name}"
    folder.mkdir(parents=True, exist_ok=True)
    for ext, data in ck.asm.items():
        product = folder / f"kernel.{ext}"
        (
            product.write_bytes(data)
            if isinstance(data, bytes)
            else product.write_text(data)
        )
    return ck


def identity(name, ck):
    return dict(
        ttir_path=str(ROOT / f"ir/{name}.ttir"),
        ttir_sha256=digest(ROOT / f"ir/{name}.ttir"),
        products={s: hash_data(v) for s, v in ck.asm.items()},
        options=ck.metadata._asdict(),
        n_regs=ck.n_regs,
        n_spills=ck.n_spills,
        shared=ck.metadata.shared,
    )


def allocate(m, n, distribution="normal"):
    x = torch.randn((m, n), device="cuda", dtype=torch.float32)
    if distribution == "constant":
        x.fill_(3.5)
    elif distribution == "offset":
        x.add_(10000)
    elif distribution == "lowvar":
        x.mul_(0.001).add_(1)
    elif distribution == "zero":
        x.zero_()
    elif distribution == "alternating":
        x[:, ::2] = 10000
        x[:, 1::2] = -10000
    elif distribution == "outlier":
        x.fill_(1)
        x[:, 0] = 10000
    vs = torch.full((m + 32,), float("nan"), device="cuda")
    ms = torch.full_like(vs, float("nan"))
    return x, vs[16:-16], ms[16:-16], vs, ms


def launch(ck, tensors, correction=1.0):
    x, var, mean = tensors[:3]
    m, n = x.shape
    assert x.dtype == torch.float32 and x.is_contiguous()
    assert 1 <= m <= 4096 and 16 <= n <= 16384 and n % 16 == 0
    assert correction in (0.0, 1.0)
    ck[(m, 1, 1)](x, var, mean, m, n, float(correction))
