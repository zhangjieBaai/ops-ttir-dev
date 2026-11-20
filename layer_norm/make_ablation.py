"""Forward progressive and leave-one-out TTIR ablations; existing artifacts stay immutable."""

import re

from common import ROOT, save_json, sha256
from make_candidates import without_locations
from make_round2 import grouped


def store_cs(text):
    return re.sub(
        r"^(\s*tt.store .*?, %v\d+)( : tensor<)",
        r"\1 cacheModifier = cs\2",
        text,
        flags=re.M,
    )


def generate():
    clean = without_locations((ROOT / "ir/baseline.ttir").read_text())
    fine = [1024, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 14336, 16384]

    def variant(limits, ca=False, cs=False, mixed=False):
        text = grouped(
            clean, 2048, limits, volatile=ca, small_chunk=1024 if mixed else None
        )
        text = text.replace("{isVolatile = true}", "cacheModifier = ca")
        return store_cs(text) if cs else text

    variants = {
        "abl_recip_only": clean.replace(
            "    %mean_5 = arith.divf %mean, %mean_4 : f32",
            "    %recip_n = arith.divf %rstd, %mean_4 : f32\n"
            "    %mean_5 = arith.mulf %mean, %recip_n : f32",
        ).replace(
            "%var_6 = arith.divf %var, %mean_4", "%var_6 = arith.mulf %var, %recip_n"
        ),
        "abl_rsqrt_only": clean.replace(
            "    %rstd_8 = math.sqrt %rstd_7 : f32\n"
            "    %rstd_9 = arith.divf %rstd, %rstd_8 : f32",
            "    %rstd_9 = math.rsqrt %rstd_7 : f32",
        ),
        "abl_arithmetic": (ROOT / "ir/arithmetic.ttir").read_text(),
        "abl_grouped": variant(None),
        "abl_unroll": variant(fine),
        "abl_ca": variant(fine, ca=True),
        "abl_cs": variant(fine, ca=True, cs=True),
        "abl_final": variant(fine, ca=True, cs=True, mixed=True),
        "abl_no_ca": variant(fine, cs=True, mixed=True),
        "abl_no_cs": variant(fine, ca=True, mixed=True),
        "abl_no_unroll": variant(None, ca=True, cs=True),
    }
    assert variants["abl_final"] == (ROOT / "ir/optimized.ttir").read_text()
    manifest = {}
    for name, text in variants.items():
        path = ROOT / f"ir/{name}.ttir"
        if path.exists():
            assert path.read_text() == text
        else:
            path.write_text(text)
        manifest[name] = sha256(path)
    save_json(ROOT / "results/ablation/candidates.json", manifest)


if __name__ == "__main__":
    generate()
