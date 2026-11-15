#!/bin/sh
# Keep a training run alive: restart it with --resume whenever it exits with
# an error (a crash, an out-of-memory kill). A clean exit ends the loop.
#
#   scripts/run_forever.sh runs/main --set budget_hours=330
set -u
run="$1"
shift
python -m cfbot.train --run "$run" "$@"
status=$?
while [ "$status" -ne 0 ]; do
    echo "training exited with status $status; resuming in 30 s" >&2
    sleep 30
    python -m cfbot.train --run "$run" --resume
    status=$?
done
