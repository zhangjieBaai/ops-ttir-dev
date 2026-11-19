"""Deterministically edit the immutable TTIR; never regenerate a Python kernel."""

import re

from common import ROOT, save_json, sha256


def memory_policy(text, load=None, store=None, load_evict=None, store_evict=None):
    result = []
    for line in text.splitlines():
        attrs = []
        if "tt.load " in line:
            if load:
                attrs.append(f"cacheModifier = {load}")
            if load_evict:
                attrs.append(f"evictionPolicy = {load_evict}")
        if "tt.store " in line:
            if store:
                attrs.append(f"cacheModifier = {store}")
            if store_evict:
                attrs.append(f"evictionPolicy = {store_evict}")
        if attrs:
            line = line.replace(" : tensor<", " " + ", ".join(attrs) + " : tensor<", 1)
        result.append(line)
    return "\n".join(result) + "\n"


def split512(text):
    start = text.index("    %offsets = tt.make_range")
    end = text.index("    tt.return", start)
    body = text[start:end]
    defined = set(re.findall(r"(%[\w]+) =", body))
    parts = []
    for suffix, lo, hi in (("lo", 0, 512), ("hi", 512, 1024)):
        chunk = re.sub(
            r"%[\w]+", lambda m: m[0] + "_" + suffix if m[0] in defined else m[0], body
        )
        chunk = re.sub(r"%([0-9]+)_" + suffix + r"\b", r"%v\1_" + suffix, chunk)
        chunk = chunk.replace("1024x", "512x")
        chunk = chunk.replace(
            "end = 1024 : i32, start = 0 : i32", f"end = {hi} : i32, start = {lo} : i32"
        )
        parts.append(chunk)
    return text[:start] + "".join(parts) + text[end:]


def main():
    text = (ROOT / "ir/baseline.ttir").read_text()
    candidates = {
        "r1_store_cs": memory_policy(text, store="cs"),
        "r1_load_cg": memory_policy(text, load="cg"),
        "r1_cg_cs": memory_policy(text, load="cg", store="cs"),
        "r1_store_evict_first": memory_policy(text, store_evict="evict_first"),
        "r1_load_last_cs": memory_policy(text, load_evict="evict_last", store="cs"),
        "r1_split512_v2": split512(text),
        "r1_split512_cs_v2": memory_policy(split512(text), store="cs"),
    }
    records = {}
    for name, value in candidates.items():
        path = ROOT / f"ir/{name}.ttir"
        if path.exists():
            assert path.read_text() == value, f"Never overwrite candidate: {path}"
        else:
            with path.open("x") as stream:
                stream.write(value)
        records[name] = dict(path=str(path.relative_to(ROOT)), sha256=sha256(path))
    path = ROOT / "results/candidates_r1_v2.json"
    save_json(
        path,
        dict(baseline_sha256=sha256(ROOT / "ir/baseline.ttir"), candidates=records),
    )
    print("Immutable candidate TTIRs ready")


if __name__ == "__main__":
    main()
