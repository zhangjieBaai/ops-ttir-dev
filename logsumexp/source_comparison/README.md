# Optimized Triton source versus optimized TTIR

The public Triton Python port reproduces the selected `r4_focus` methods.
[Report](report.md) and [all-shape times](results/all_shapes.csv) compare the
native source path with the immutable TTIR file path on the original30 shapes.
Original baseline, selected TTIR and FlagGems code are unchanged.

Direct TTIR/source geometric ratios: 0.99941x main, 0.99889x independent repeat.
Ratios above1 mean source faster. Overall times are close; individual cases
differ by about1–3%. The report separates observations from causal hypotheses.
Source and TTIR outputs match bitwise for all408 inputs in each seed.

Run from `/root/ops-ttir-dev`, using a checked idle GPU and a new output path:

```bash
CUDA_VISIBLE_DEVICES=0 /root/venvs/ir-ops-dev/bin/python \
  logsumexp/source_comparison/compare.py --output /root/tmp/source-comparison-new \
  --rounds 7 --seed 42
```

`--accuracy-only` skips timing. `--sanitizer` tests every51 shapes with normal
and all-negative-infinity inputs; use compute-sanitizer with the logsumexp
kernel filter. Saved artifacts live under the existing ignored
`logsumexp/artifacts/source_comparison/`; recorded stage hashes and code can
rebuild them. Pilot harness failures remain classified in results/pilot_failures.json.
No Python algorithm change was needed after the first source port; all repairs
were to source/file launcher adaptation or evidence handling before timing.
