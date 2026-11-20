#!/bin/bash
# Run phasic_single_run.py (save mode) on every subject/run that has an
# eye-tracking physio file and an events file.
# Usage: bash run_phasic_all.sh [covariate]   (default: vtc)
# Writes derivatives/eye_tracking/phasic/batch_status_<cov>.tsv (subj, run, status, message).
cov=${1:-vtc}
project=/projectnb/nphfnirs/s/datasets/gradCPT_NN24
py=${PYTHON:-/projectnb/stephenlab/cchang1/.conda/envs/cedalion/bin/python}
here=$(cd "$(dirname "$0")" && pwd)
outdir=$project/derivatives/eye_tracking/phasic
mkdir -p "$outdir"
status=$outdir/batch_status_${cov}.tsv
printf 'subj\trun\tstatus\tmessage\n' > "$status"
cd "$project"
for s in $(ls -d sub-*); do
  for r in 1 2 3; do
    rr=$(printf '%02d' $r)
    ls $s/nirs/${s}_task-gradCPT_run-${rr}_recording-eyetracking_physio*.tsv >/dev/null 2>&1 || continue
    [ -f $s/nirs/${s}_task-gradCPT_run-${rr}_events.tsv ] || continue
    echo "=== $s run-$rr ($(date +%T))"
    log=$("$py" "$here/phasic_single_run.py" "$s" "$r" "$cov" save 2>&1)
    if [ $? -eq 0 ]; then
      printf '%s\t%d\tok\t\n' "$s" "$r" >> "$status"
    else
      msg=$(echo "$log" | grep -v '^\s*$' | tail -1 | tr '\t' ' ')
      printf '%s\t%d\tfailed\t%s\n' "$s" "$r" "$msg" >> "$status"
      echo "  FAILED: $msg"
    fi
  done
done
echo "done: $status"
