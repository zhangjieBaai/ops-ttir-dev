#!/usr/bin/env bash
# New evidence root for every full comparison; no source or old baseline changes.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Choose one idle physical H100 GPU}"
cd "$(dirname "$0")/../.."
RUN=$(mktemp -d mm/comparison/results/replay_XXXXXX)
mkdir "$RUN/results"
cp mm/comparison/README.md "$RUN/README.md"
exec > >(tee "$RUN/run.log") 2>&1
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=""
export USE_C_EXTENSION=0
export FLAGGEMS_CACHE_DIR="$PWD/$RUN/cache/flaggems"
export TRITON_CACHE_DIR="$PWD/$RUN/cache/triton"
unset PYTHONPATH
PY=/root/venvs/ir-ops-dev/bin/python
FT=/usr/bin/python
W=mm/comparison/worker.py
R="$RUN/results"
trap 'status=$?; echo "exit_status=$status" > "$RUN/exit_status.txt"' EXIT
set -x
PYTHONPATH=/root/FlagGems/src "$FT" "$W" flagtree --output "$R/flagtree_main"
"$PY" "$W" upstream --output "$R/upstream_main"
PYTHONPATH=/root/FlagGems/src "$FT" "$W" flagtree --seed 123 --frozen "$R/flagtree_main/summary.json" --output "$R/flagtree_repeat"
"$PY" "$W" upstream --seed 123 --output "$R/upstream_repeat"
for phase in main repeat; do
    seed=20261119
    [[ "$phase" == repeat ]] && seed=123
    mapfile -t sizes < <("$PY" - "$R/flagtree_$phase/summary.json" <<'PY'
import json,sys
for row in json.load(open(sys.argv[1]))['records']:
    if not row['valid']:
        print(row['n'])
PY
)
    if (( ${#sizes[@]} )); then
        PYTHONPATH=/root/FlagGems/src "$FT" "$W" flagtree --seed "$seed" --rounds 7 --sizes "${sizes[@]}" \
            --frozen "$R/flagtree_main/summary.json" --output "$R/repair_$phase"
    else
        "$PY" - "$R/repair_$phase" "$seed" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);p.mkdir()
(p/'summary.json').write_text(json.dumps(dict(passed=True,seed=int(sys.argv[2]),rounds=7,records=[],
    status='NOT_APPLICABLE',reason='No originally invalid points; no repair GPU workload executed'),indent=2)+'\n')
PY
    fi
 done
"$FT" mm/comparison/analyze.py --results "$R" --output-dir "$RUN"
echo "Comparison report: $RUN/report.md"
