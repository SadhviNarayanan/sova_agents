#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${SOVA_AGENTS_URL:-https://sova-agents.onrender.com}"
PATIENT_ID="${1:-${PATIENT_ID:-CR-003}}"
OPENAPI_TMP="$(mktemp)"
STATUS_TMP="$(mktemp)"
trap 'rm -f "$OPENAPI_TMP" "$STATUS_TMP"' EXIT

green() { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
red() { printf '\033[31m%s\033[0m\n' "$*"; }
muted() { printf '\033[2m%s\033[0m\n' "$*"; }

echo "Checking Sova agents deploy"
muted "Service: $BASE_URL"
muted "Patient: $PATIENT_ID"
echo

openapi_code="$(
  curl -sS --retry 3 --retry-delay 5 --retry-all-errors \
    --max-time 20 \
    -w '%{http_code}' \
    -o "$OPENAPI_TMP" \
    "$BASE_URL/openapi.json" || true
)"

if [[ "$openapi_code" != "200" ]]; then
  red "Deploy check failed: service did not return OpenAPI. HTTP $openapi_code"
  echo
  cat "$OPENAPI_TMP" || true
  exit 1
fi

if ! python3 - "$OPENAPI_TMP" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    api = json.load(f)

paths = set(api.get("paths", {}))
required = {
    "/v1/patients/{patient_id}/status",
    "/start-debate/{patient_id}",
    "/stream/{patient_id}",
}
missing = sorted(required - paths)
if missing:
    print("Missing routes:", ", ".join(missing))
    raise SystemExit(1)
PY
then
  red "Deploy check failed: latest routes are not live yet."
  exit 1
fi

green "Deploy is live: required routes are present."
echo

status_code="$(
  curl -sS --retry 1 --retry-delay 2 --max-time 35 \
    -w '%{http_code}' \
    -o "$STATUS_TMP" \
    "$BASE_URL/v1/patients/$PATIENT_ID/status" || true
)"

case "$status_code" in
  200)
    green "Patient status is healthy."
    python3 - "$STATUS_TMP" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

vitals = data.get("vitals", {})
print(f"riskLevel: {data.get('riskLevel')}")
print(f"anomalyLevel: {data.get('anomalyLevel')}")
print(f"recommendedAction: {data.get('recommendedAction')}")
print(
    "vitals:",
    f"heartRate={vitals.get('heartRate')}",
    f"spo2={vitals.get('spo2')}",
    f"temperature={vitals.get('temperature')}",
    f"bloodPressure={vitals.get('bloodPressure')}",
    f"timestamp={vitals.get('timestamp')}",
)
PY
    ;;
  404)
    yellow "Deploy is up, but no vitals row exists for patient '$PATIENT_ID'."
    cat "$STATUS_TMP"
    ;;
  503)
    red "Deploy is up, but backend cannot read vitals from BigQuery."
    muted "Most likely: Render is missing GOOGLE_APPLICATION_CREDENTIALS_JSON or BigQuery table/query config is wrong."
    cat "$STATUS_TMP"
    exit 2
    ;;
  000)
    red "Patient status request timed out or could not connect."
    exit 3
    ;;
  *)
    red "Patient status returned HTTP $status_code"
    cat "$STATUS_TMP"
    exit 4
    ;;
esac

