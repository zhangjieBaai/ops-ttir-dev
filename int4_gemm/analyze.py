"""Recompute paired summaries, bootstrap uncertainty and compiled-code evidence."""

import json
import math
import random
import statistics
import struct

from common import ROOT, digest, manifest, save, sha


def elf_text(path):
    blob = path.read_bytes()
    assert blob[:6] == b"\x7fELF\x02\x01"
    offset = struct.unpack_from("<Q", blob, 40)[0]
    size, count, names_index = struct.unpack_from("<HHH", blob, 58)
    headers = [
        struct.unpack_from("<IIQQQQIIQQ", blob, offset + i * size) for i in range(count)
    ]
    names = headers[names_index]
    names = blob[names[4] : names[4] + names[5]]
    sections = {}
    for header in headers:
        name = names[header[0] :].split(b"\0", 1)[0].decode()
        if name.startswith(".text"):
            sections[name] = digest(blob[header[4] : header[4] + header[5]])
    return sections


def summarize(path):
    data = json.loads(path.read_text())
    names = sorted(set(row["name"] for row in data["results"]))
    result = {}
    rng = random.Random(2026)
    for name in names:
        rows = [r for r in data["results"] if r["name"] == name]
        values = [r["median_speedup"] for r in rows]
        bootstraps = []
        for _ in range(3000):
            logs = []
            for row in rows:
                samples = [s["speedup"] for s in row["samples"]]
                logs.append(
                    math.log(statistics.median(rng.choices(samples, k=len(samples))))
                )
            bootstraps.append(math.exp(statistics.mean(logs)))
        bootstraps.sort()
        result[name] = dict(
            geomean=math.exp(statistics.mean(math.log(v) for v in values)),
            arithmetic=statistics.mean(values),
            minimum=min(values),
            maximum=max(values),
            bootstrap95_fixed_shapes=[bootstraps[75], bootstraps[2924]],
        )
    return result


def main():
    output = dict(
        summaries={
            run: summarize(ROOT / "results" / (run + ".json"))
            for run in ("screen", "main", "repeat")
        },
        code=[],
    )
    for case in manifest()["cases"]:
        base = ROOT / "artifacts" / case["id"] / "baseline" / "kernel.cubin"
        cast = ROOT / "artifacts" / case["id"] / "cast_after_v1" / "kernel.cubin"
        output["code"].append(
            dict(
                case=case["id"],
                baseline_text=elf_text(base),
                cast_after_text=elf_text(cast),
                cast_after_text_equal=elf_text(base) == elf_text(cast),
                baseline_cubin_sha256=sha(base),
                cast_after_cubin_sha256=sha(cast),
            )
        )
    save(ROOT / "results/analysis.json", output)
    print(json.dumps(output["summaries"], indent=2))


if __name__ == "__main__":
    main()
