#!/usr/bin/env bash
# Runs every scenario against a running server, writes stress/results/<date>.md.
set -euo pipefail
cd "$(dirname "$0")"
out="results/$(date +%F).md"
{ echo "# Stress run $(date -Iseconds)"; echo; echo "Host: $(uname -srm), $(getconf _NPROCESSORS_ONLN) vCPU"; echo; } > "$out"
# Arrival rates (req/s). cold=3 is inside M1 Pro capacity (~3 cold req/s, see results/2026-09-21-m1pro.md);
# mixed/batch/overload defaults deliberately exceed it, so their k6 thresholds fail on 529s by design.
rate() {
  case "$1" in
    cold) echo "${COLD_RATE:-3}" ;;
    hot) echo "${HOT_RATE:-3000}" ;;
    mixed) echo "${MIXED_RATE:-60}" ;;
    batch) echo "${BATCH_RATE:-5}" ;;
    overload) echo "${OVERLOAD_RATE:-100}" ;;
  esac
}
for s in cold hot mixed batch overload; do
  echo "## $s ($(rate "$s") req/s)" >> "$out"
  k6 run --quiet -e SCENARIO="$s" -e RATE="$(rate "$s")" --summary-trend-stats "avg,p(50),p(95),p(99),max" scenario.js 2>&1 \
    | grep -E 'http_req_duration|http_reqs|http_req_failed|checks|dropped_iterations' | sed 's/^/    /' >> "$out" || echo "    (thresholds failed)" >> "$out"
  curl -s "${RSD_METRICS:-http://127.0.0.1:9000}/metrics" | grep -E '^rsdecider_(batch_size|pending|cache_total|shed_total|coalesced_total)' \
    | sed 's/^/    /' >> "$out" || true
  echo >> "$out"
done
echo "wrote $out"
