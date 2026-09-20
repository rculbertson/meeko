#!/usr/bin/env bash
# Cyclomatic-complexity gate. Invoked identically by CI and the pre-push hook,
# so the thresholds live here once rather than drifting between the two.
#
#   --max-absolute C    no block in meeko/ above CC 20 (radon rank D or worse)
#   --max-average-num   the package mean, pinned just above its current value
#
# The numeric average is the ratchet. Xenon's letter-grade --max-average A
# would be near-inert here: rank A runs to CC 5.0 and the package sits at
# 2.82, so 42 more CC-20 functions -- every one passing --max-absolute C --
# would still leave the build green. Re-pin AVERAGE downward as the real
# average improves; that is what makes it a ratchet rather than a ceiling.
#
# tests/ is out of scope: linear setup-then-assert scores badly for reasons
# that don't signal a maintenance problem.
set -uo pipefail

ABSOLUTE=C
AVERAGE=3.1
TARGET=meeko

out=$(uv run xenon --max-absolute "$ABSOLUTE" --max-average-num "$AVERAGE" "$TARGET" 2>&1)
status=$?
[ -n "$out" ] && printf '%s\n' "$out"

# A module radon cannot parse is logged and skipped -- it drops out of both
# the absolute and the average check without affecting xenon's exit code.
# Silent partial coverage is worse than a failure, so treat it as one.
if printf '%s' "$out" | grep -q 'cannot parse'; then
  echo "error: a module was skipped by the complexity gate (see above)" >&2
  exit 1
fi

exit $status
