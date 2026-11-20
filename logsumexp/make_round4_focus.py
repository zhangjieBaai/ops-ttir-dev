"""Apply measured mechanisms only to their recorded algorithmic regimes."""

from common import ROOT, ir_path, manifest, save_json, sha256
from make_candidates import narrow_indices
from make_round4 import row_group
from make_round4_lane import lane32


def focus(text, config):
    if "TILE_K" in config["meta"]:
        return narrow_indices(text)
    if config["meta"]["ONE_TILE_PER_CTA"] and config["meta"]["TILE_N"] <= 128:
        return (
            lane32(text, config)
            if "%M:" not in text
            else row_group(text, config, 8, 128)
        )
    return text


def main():
    result = {}
    for group, config in manifest()["groups"].items():
        parent = ir_path(group, "r3_mix")
        path = ir_path(group, "r4_focus")
        with path.open("x") as stream:
            stream.write(focus(parent.read_text(), config))
        result[group] = dict(sha256=sha256(path), parent_sha256=sha256(parent))
    save_json(
        ROOT / "results/runs/r4_adaptive/candidate_focus.json",
        dict(
            groups=result,
            generator_sha256=sha256(__file__),
            reason="Valid5round screen: restrict row8/lane32 to tile<=128. "
            "i32 non-inner small gains without severe regressions. Host mapping stays frozen.",
        ),
    )


if __name__ == "__main__":
    main()
