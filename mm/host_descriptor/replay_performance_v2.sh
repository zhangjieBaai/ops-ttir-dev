#!/usr/bin/env bash
# User exception permits warnings, not numerical/memory failures or fake PASS.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Choose one idle H100 GPU}"
cd "$(dirname "$0")/../.."
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=""
PY=/root/venvs/ir-ops-dev/bin/python
REG=mm/host_descriptor/results/prototype_inputs_v4.json
RUN=$(mktemp -d mm/host_descriptor/results/performance_v2/replay_XXXXXX)
exec > >(tee "$RUN/run.log") 2>&1
set -x
"$PY" mm/host_descriptor/probe.py --full --names envelope_baseline envelope_host_v1 host_pid_v2 --registry "$REG" --output "$RUN/check"
compute-sanitizer --tool memcheck --error-exitcode 99 --log-file "$RUN/memcheck.log" \
    "$PY" mm/host_descriptor/probe.py --full --names host_pid_v2 --registry "$REG" --output "$RUN/memcheck_check"
set +e
compute-sanitizer --tool racecheck --error-exitcode 99 --log-file "$RUN/racecheck.log" \
    "$PY" mm/host_descriptor/probe.py --full --names host_pid_v2 --registry "$REG" --output "$RUN/racecheck_check"
race_status=$?
set -e
printf '%s\n' "$race_status" > "$RUN/racecheck_exit_status.txt"
# Whitelist only the exact unresolved warning pattern authorized by the user.
"$PY" - "$RUN/racecheck.log" "$RUN/racecheck_check/summary.json" "$race_status" <<'PY'
import json,re,sys
text=open(sys.argv[1]).read();data=json.load(open(sys.argv[2]));status=int(sys.argv[3])
assert data['passed']
assert status in (0,99)
match=re.search(r'RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)',text)
assert match and int(match[2])==0
if status==99:
    assert int(match[1])==int(match[3])==37, 'Unexpected tool finding is not covered by the waiver'
print('Actual race status retained; synchronization remains unqualified')
PY
"$PY" mm/host_descriptor/performance_v2.py --validation "$RUN/check/summary.json" --output "$RUN/main.json"
"$PY" mm/host_descriptor/probe.py --full --seed 123 --names envelope_baseline envelope_host_v1 host_pid_v2 --registry "$REG" --output "$RUN/repeat_check"
"$PY" mm/host_descriptor/performance_v2.py --seed 123 --rounds 5 --validation "$RUN/repeat_check/summary.json" --output "$RUN/repeat.json"
"$PY" mm/host_descriptor/check_performance_entry.py --output "$RUN/public_entry.json"
echo "Paired evidence: $RUN (user race warning exception; no automatic selection rewrite)"
