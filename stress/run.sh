#!/usr/bin/env bash
# Runs every scenario against a running server, writes stress/results/<date>.md.
set -euo pipefail
cd "$(dirname "$0")"
out="results/$(date +%F).md"
{ echo "# Stress run $(date -Iseconds)"; echo; echo "Host: $(uname -srm), $(getconf _NPROCESSORS_ONLN) vCPU"; echo; } > "$out"
for s in cold hot mixed batch overload; do
  echo "## $s" >> "$out"
  k6 run --quiet -e SCENARIO="$s" --summary-trend-stats "avg,p(50),p(95),p(99),max" scenario.js 2>&1 \
    | grep -E 'http_req_duration|http_reqs|http_req_failed|checks' | sed 's/^/    /' >> "$out" || echo "    (thresholds failed)" >> "$out"
  curl -s "${RSD_METRICS:-http://127.0.0.1:9000}/metrics" | grep -E '^rsdecider_(batch_size|pending|cache_total|shed_total|coalesced_total)' \
    | sed 's/^/    /' >> "$out" || true
  echo >> "$out"
done
echo "wrote $out"
