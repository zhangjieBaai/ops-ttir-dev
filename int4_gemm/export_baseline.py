"""Limited prespecified configuration screen, followed by baseline freeze."""

import statistics
import subprocess

import torch
import triton
from common import (
    CONFIGS,
    ROOT,
    SHAPES,
    allocate,
    assert_output,
    case_id,
    environment,
    record,
    reference,
    save,
    sha,
    source_launch,
    write_once,
)


def main():
    env = environment()
    assert not (ROOT / "baseline.json").exists()
    cases = []
    torch.backends.cuda.matmul.allow_tf32 = False
    for shape in SHAPES:
        ts = allocate(shape)
        ref = reference(ts)
        trials = []
        for config in CONFIGS:
            ck = source_launch(ts, config)
            error = assert_output(ts, ref)
            times = [
                triton.testing.do_bench_cudagraph(
                    lambda: source_launch(ts, config), rep=10
                )
                for _ in range(3)
            ]
            trials.append(dict(config=config, ms=times, max_abs_error=error))
        selected = min(trials, key=lambda t: statistics.median(t["ms"]))["config"]
        ck = source_launch(ts, selected)
        cid = case_id(shape)
        path = ROOT / "ir" / cid / "baseline.ttir"
        write_once(path, ck.asm["ttir"])
        cases.append(
            dict(
                id=cid,
                shape=shape,
                config=selected,
                trials=trials,
                source=record(ck, path),
            )
        )
        print(cid, selected, flush=True)
    save(
        ROOT / "baseline.json",
        dict(
            cases=cases,
            environment=env,
            source=dict(
                path="upstream/source.py.snapshot",
                sha256=sha(ROOT / "upstream/source.py.snapshot"),
                extracted_sha256=sha(ROOT / "tutorial.py"),
                flaggems_commit=subprocess.check_output(
                    ["git", "-C", "/root/FlagGems", "rev-parse", "HEAD"], text=True
                ).strip(),
                extraction=(
                    "Complete kernel; removed libentry, tle alias points to upstream tl; "
                    "no computational changes."
                ),
            ),
        ),
    )


if __name__ == "__main__":
    main()
