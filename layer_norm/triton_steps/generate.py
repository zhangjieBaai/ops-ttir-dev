"""Build readable Triton source witnesses; never replace the TTIR experiment."""

import ast
import difflib
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OP = ROOT.parent
LIMITS = [1024, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 14336, 16384]
DESCRIPTIONS = {
    "baseline": "Original algorithm: 8192-wide dynamic loops, divisions, sqrt.",
    "reciprocal_only": "Only reuse 1/N for the mean and variance.",
    "rsqrt_only": "Only replace 1/sqrt by rsqrt.",
    "arithmetic": "Step 1: reciprocal reuse plus rsqrt.",
    "grouped": "Step 2: keep step 1 and use 2048-wide dynamic loops.",
    "unroll": "Step 3: keep step 2 and statically unroll by runtime N ranges.",
    "mean_ca": "Step 4: keep step 3 and mark only mean-stage input loads .ca.",
    "output_cs": "Step 5: keep step 4 and mark only output Y stores .cs.",
    "mixed": "Step 6: keep step 5 and use 1024-wide blocks when N <= 3072.",
}


def reciprocal(text):
    return text.replace(
        "mean = tl.sum(_mean, axis=0) / N",
        "inv_n = 1.0 / N\n    mean = tl.sum(_mean, axis=0) * inv_n",
    ).replace("var = tl.sum(_var, axis=0) / N", "var = tl.sum(_var, axis=0) * inv_n")


def rsqrt(text):
    return text.replace("rstd = 1 / tl.sqrt(var + eps)", "rstd = tl.rsqrt(var + eps)")


def mean_ca(text):
    return re.sub(
        r"(a = tl.load\(X \+ cols, mask=cols < N, other=0\.0)(\))",
        r'\1, cache_modifier=".ca"\2',
        text,
    )


def output_cs(text):
    return text.replace(
        "tl.store(Y + cols, y, mask=mask)",
        'tl.store(Y + cols, y, mask=mask, cache_modifier=".cs")',
    )


def dispatch(body, mixed=False):
    dynamic = body.replace("def kernel(", "def _dynamic_body(", 1)
    static = body.replace("def kernel(", "def _static_body(", 1)
    static = static.replace(
        "BLOCK_SIZE: tl.constexpr", "BLOCK_SIZE: tl.constexpr, K: tl.constexpr"
    )
    static = static.replace(
        "for off in range(0, N, BLOCK_SIZE):", "for block in tl.static_range(K):"
    )
    static = static.replace(
        "cols = off + tl.arange(0, BLOCK_SIZE)",
        "cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
    )
    args = "X, Y, W, B, Mean, Rstd, stride, N, eps"
    wrapper = f"@triton.jit\ndef kernel({args}, BLOCK_SIZE: tl.constexpr):\n"
    for i, limit in enumerate(LIMITS):
        width = 1024 if mixed and limit <= 3072 else 2048
        count = (limit + width - 1) // width
        wrapper += f"    {'if' if i == 0 else 'elif'} N <= {limit}:\n"
        wrapper += f"        _static_body({args}, {width}, {count})\n"
    wrapper += f"    else:\n        _dynamic_body({args}, 2048)\n"
    return dynamic + "\n\n" + static + "\n\n" + wrapper


def source(name, body, width):
    return (
        f'"""{DESCRIPTIONS[name]}"""\n\n'
        "import triton\nimport triton.language as tl\n\n"
        f"BLOCK_SIZE = {width}\n\n" + body + "\n"
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(path, text):
    if path.exists():
        assert (
            path.read_text() == text
        ), f"Refusing to overwrite {path}; use a new version"
    else:
        path.write_text(text)


def main():
    tutorial = OP / "tutorial.py"
    parsed = ast.parse(tutorial.read_text())
    fn = next(n for n in parsed.body if isinstance(n, ast.FunctionDef))
    fn.name = "kernel"
    base = ast.unparse(fn)
    arithmetic = rsqrt(reciprocal(base))
    unroll = dispatch(arithmetic)
    bodies = {
        "baseline": (base, 8192),
        "reciprocal_only": (reciprocal(base), 8192),
        "rsqrt_only": (rsqrt(base), 8192),
        "arithmetic": (arithmetic, 8192),
        "grouped": (arithmetic, 2048),
        "unroll": (unroll, 2048),
        "mean_ca": (mean_ca(unroll), 2048),
        "output_cs": (output_cs(mean_ca(unroll)), 2048),
        "mixed": (output_cs(mean_ca(dispatch(arithmetic, mixed=True))), 2048),
    }
    for name, (body, width) in bodies.items():
        path = ROOT / f"{name}.py"
        formatted = subprocess.run(
            ["python", "-m", "black", "--quiet", "-"],
            input=source(name, body, width),
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        freeze(path, formatted)
    comparisons = [("baseline", "reciprocal_only"), ("baseline", "rsqrt_only")]
    chain = [
        "baseline",
        "arithmetic",
        "grouped",
        "unroll",
        "mean_ca",
        "output_cs",
        "mixed",
    ]
    comparisons += list(zip(chain, chain[1:]))
    diffs = ROOT / "diffs"
    diffs.mkdir(exist_ok=True)
    for before, after in comparisons:
        for reference in dict.fromkeys((before, "baseline")):
            output = diffs / f"{reference}_to_{after}.diff"
            freeze(
                output,
                "".join(
                    difflib.unified_diff(
                        (ROOT / f"{reference}.py")
                        .read_text()
                        .splitlines(keepends=True),
                        (ROOT / f"{after}.py").read_text().splitlines(keepends=True),
                        fromfile=f"{reference}.py",
                        tofile=f"{after}.py",
                    )
                ),
            )
    manifest = dict(
        tutorial_sha256=sha(tutorial),
        purpose="Readable source witnesses, no TTIR promotion",
        variants={
            name: dict(
                source=f"{name}.py",
                sha256=sha(ROOT / f"{name}.py"),
                block_size=width,
                description=DESCRIPTIONS[name],
            )
            for name, (_, width) in bodies.items()
        },
    )
    freeze(ROOT / "manifest.json", json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
