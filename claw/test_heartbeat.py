"""
test_heartbeat.py — offline unit + integration tests for the heartbeat loop.

Run from repo root:
    python -m pytest claw/test_heartbeat.py -v

Or from claw/:
    python -m pytest test_heartbeat.py -v
    python test_heartbeat.py          # unittest runner
"""

from __future__ import annotations

import sys
import types
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

# ── stub heavy/missing deps BEFORE any project imports ───────────────────────
# Prevents ImportError for google-cloud, twilio, db (no db.py exists yet), etc.

def _stub(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)
    return mod

_stub("db",          log_anomaly_to_db=MagicMock())
_stub("big_query",   get_anchor=MagicMock(), fetch_by_timestamp=MagicMock())
_stub("query",       get_patient_profile=MagicMock())
_stub("call_twilio", call_911=MagicMock(), call_caregiver=MagicMock(), text_caregiver=MagicMock())
_stub("call_server", register_call_data=MagicMock())

for _pkg in ("google", "google.cloud", "google.cloud.bigquery", "twilio", "twilio.rest"):
    _stub(_pkg, Client=MagicMock())

# Make google.cloud.bigquery.Client accessible via attribute chain
sys.modules["google"].cloud          = sys.modules["google.cloud"]
sys.modules["google.cloud"].bigquery = sys.modules["google.cloud.bigquery"]

sys.path.insert(0, str(Path(__file__).resolve().parent))

import heartbeat  # noqa: E402
from anomaly import _fallback_rule_based_level  # noqa: E402


# ── shared fixtures ───────────────────────────────────────────────────────────

_ANCHOR = datetime(2026, 4, 1, 6, 0, 0)

# Synthetic patient snapshots provided as seed data.
# "current" = the abnormal reading that should trigger escalation.
# "history" = values near each patient's baseline (normal prior ticks).
_PAT_L4_002_CURRENT = {
    "user_id": "PAT-SYN-L4-002", "date": "2026-04-26T00:00:00",
    "recovery_score": 16.0, "day_strain": 18.4,
    "hrv": 28.0, "hrv_baseline": 54.0,
    "resting_heart_rate": 99.0, "rhr_baseline": 74.0,
    "respiratory_rate": 21.0, "skin_temp_deviation": 1.4,
    "sleep_performance": 48.0,
}
_PAT_L4_002_HISTORY = {  # near-baseline — should score 0–1
    "user_id": "PAT-SYN-L4-002", "date": "2026-04-20T00:00:00",
    "recovery_score": 68.0, "day_strain": 9.0,
    "hrv": 52.0, "hrv_baseline": 54.0,
    "resting_heart_rate": 76.0, "rhr_baseline": 74.0,
    "respiratory_rate": 15.5, "skin_temp_deviation": 0.1,
    "sleep_performance": 74.0,
}

_PAT_L3_001_CURRENT = {
    "user_id": "PAT-SYN-L3-001", "date": "2026-04-29T00:00:00",
    "recovery_score": 29.0, "day_strain": 16.2,
    "hrv": 37.0, "hrv_baseline": 58.0,
    "resting_heart_rate": 90.0, "rhr_baseline": 74.0,
    "respiratory_rate": 19.6, "skin_temp_deviation": 0.9,
    "sleep_performance": 56.0,
}
_PAT_L3_001_HISTORY = {  # near-baseline — should score 0–1
    "user_id": "PAT-SYN-L3-001", "date": "2026-04-22T00:00:00",
    "recovery_score": 71.0, "day_strain": 10.5,
    "hrv": 56.0, "hrv_baseline": 58.0,
    "resting_heart_rate": 75.0, "rhr_baseline": 74.0,
    "respiratory_rate": 15.0, "skin_temp_deviation": 0.2,
    "sleep_performance": 78.0,
}


def _profile(severity: int = 1, stage: int = 2) -> dict:
    discharge = (date.today() - timedelta(days=14)).isoformat()
    return {
        "patientId":             "TEST-001",
        "Age":                   65,
        "Gender":                "Male",
        "Surgery":               "CABG",
        "DischargeDate":         discharge,
        "RiskLevel":             ["low", "medium", "high"][severity],
        "BloodPressure":         "130/80",
        "HeartRate":             72,
        "Allergies":             "Penicillin",
        "CurrentMedications":    "Metoprolol, Aspirin",
        "EmergencyContactName":  "Linda Hartley",
        "EmergencyContactPhone": "+15125550192",
        "severity":              severity,
        "stage":                 stage,
    }


def _whoop(
    *,
    recovery:       float = 75,
    hrv:            float = 60,
    hrv_baseline:   float = 60,
    rhr:            float = 58,
    rhr_baseline:   float = 58,
    respiratory:    float = 14,
    skin_temp_dev:  float = 0.1,
    sleep_perf:     float = 80,
    day_strain:     float = 10,
    user_id:        str   = "TEST-001",
    ts:             str   = "2026-04-25T06:00:00",
) -> dict:
    return {
        "user_id":               user_id,
        "date":                  ts,
        "recovery_score":        recovery,
        "hrv":                   hrv,
        "hrv_baseline":          hrv_baseline,
        "resting_heart_rate":    rhr,
        "rhr_baseline":          rhr_baseline,
        "respiratory_rate":      respiratory,
        "skin_temp_deviation":   skin_temp_dev,
        "sleep_performance":     sleep_perf,
        "day_strain":            day_strain,
    }


# ── 1. calculate_polling_freq ─────────────────────────────────────────────────

class TestCalculatePollingFreq(TestCase):

    def test_minimum_floor_enforced(self):
        self.assertGreaterEqual(heartbeat.calculate_polling_freq(0, 0), 0.25)

    def test_higher_severity_shorter_or_equal_interval(self):
        low  = heartbeat.calculate_polling_freq(0, 0)
        high = heartbeat.calculate_polling_freq(2, 0)
        self.assertLessEqual(high, low)

    def test_later_stage_longer_or_equal_interval(self):
        early = heartbeat.calculate_polling_freq(1, 0)
        late  = heartbeat.calculate_polling_freq(1, 5)
        self.assertGreaterEqual(late, early)

    def test_known_floor_case(self):
        # severity=2 stage=0: (1*0.45)+(1*0.55)=1.0
        self.assertAlmostEqual(heartbeat.calculate_polling_freq(2, 0), 1.0)

    def test_known_no_floor_case(self):
        # severity=2 stage=5: (1*0.45)+(5*0.55)=3.2
        self.assertAlmostEqual(heartbeat.calculate_polling_freq(2, 5), 3.2)

    def test_all_severity_stage_combos_are_positive(self):
        for sev in range(3):
            for stg in range(6):
                self.assertGreater(heartbeat.calculate_polling_freq(sev, stg), 0)


# ── 2. fallback anomaly scoring ───────────────────────────────────────────────

class TestFallbackAnomalyLevel(TestCase):
    """Tests the rule-based path of infer_anomaly_level (no ML model needed)."""

    def test_healthy_vitals_level_0(self):
        snap = _whoop(recovery=80, hrv=65, hrv_baseline=65, rhr=55, rhr_baseline=55)
        self.assertEqual(_fallback_rule_based_level(snap), 0)

    def test_mild_anomaly_at_least_level_1(self):
        # recovery <50 (+1) and hrv ratio 0.77 (no penalty), rhr_delta=+4 (no penalty)
        snap = _whoop(recovery=45, hrv=50, hrv_baseline=65, rhr=59, rhr_baseline=55)
        self.assertGreaterEqual(_fallback_rule_based_level(snap), 1)

    def test_moderate_anomaly_at_least_level_2(self):
        # recovery <35 (+2), hrv ratio=0.65 (<0.70, +2), rhr_delta=+8 (+1) → 5pts → level 2
        snap = _whoop(recovery=30, hrv=42, hrv_baseline=65, rhr=63, rhr_baseline=55)
        self.assertGreaterEqual(_fallback_rule_based_level(snap), 2)

    def test_severe_anomaly_at_least_level_3(self):
        # recovery <35 (+2), hrv <0.55 ratio (+3), rhr_delta=+15 (+2), sleep<55 (+1) → 8pts → level 3
        snap = _whoop(recovery=25, hrv=30, hrv_baseline=65, rhr=70, rhr_baseline=55, sleep_perf=50)
        self.assertGreaterEqual(_fallback_rule_based_level(snap), 3)

    def test_critical_fast_path_level_4(self):
        # recovery <20 AND rhr_delta ≥20 → short-circuit to level 4
        snap = _whoop(recovery=15, hrv=40, hrv_baseline=65, rhr=80, rhr_baseline=55)
        self.assertEqual(_fallback_rule_based_level(snap), 4)

    def test_high_skin_temp_escalates(self):
        snap = _whoop(recovery=30, hrv=42, hrv_baseline=65, rhr=63, rhr_baseline=55, skin_temp_dev=1.5)
        self.assertGreaterEqual(_fallback_rule_based_level(snap), 2)

    def test_missing_fields_does_not_raise(self):
        level = _fallback_rule_based_level({})
        self.assertIn(level, range(5))

    def test_output_always_in_range(self):
        for snap in [
            _whoop(recovery=100, hrv=80, hrv_baseline=80),
            _whoop(recovery=0,   hrv=0,  hrv_baseline=80, rhr=120, rhr_baseline=55),
        ]:
            level = _fallback_rule_based_level(snap)
            self.assertIn(level, range(5), msg=f"level {level} out of range for snap {snap}")


# ── 3. run() loop ─────────────────────────────────────────────────────────────

class TestHeartbeatRun(TestCase):
    """
    Runs heartbeat.run() for one tick with all external I/O mocked.
    _fire is patched to be synchronous so mock assertions are deterministic.
    """

    def _run(self, snap: dict, profile: dict | None = None, *, mock_debate=True):
        """Run one tick and return a dict of action mocks."""
        profile = profile or _profile()
        mocks = {k: MagicMock() for k in ("debate", "call_911", "call_caregiver",
                                           "text_caregiver", "log_anomaly")}
        patches = [
            patch.object(heartbeat, "get_patient_profile", return_value=profile),
            patch.object(heartbeat, "get_anchor",          return_value=_ANCHOR),
            patch.object(heartbeat, "fetch_by_timestamp",  return_value=snap),
            # make _fire synchronous so we don't race against daemon threads
            patch.object(heartbeat, "_fire", side_effect=lambda fn, *a: fn(*a)),
            patch.object(heartbeat, "call_911",          mocks["call_911"]),
            patch.object(heartbeat, "call_caregiver",    mocks["call_caregiver"]),
            patch.object(heartbeat, "text_caregiver",    mocks["text_caregiver"]),
            patch.object(heartbeat, "log_anomaly_to_db", mocks["log_anomaly"]),
            patch.object(heartbeat.time, "sleep", return_value=None),
        ]
        if mock_debate:
            patches.append(patch.object(heartbeat, "_trigger_debate", mocks["debate"]))

        # Python ≥3.10: use ExitStack to apply a variable list of patches
        import contextlib
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            heartbeat.run("TEST-001", max_ticks=1)

        return mocks

    def test_normal_vitals_skip_debate(self):
        snap = _whoop(recovery=80, hrv=65, hrv_baseline=65, rhr=55, rhr_baseline=55)
        mocks = self._run(snap)
        mocks["debate"].assert_not_called()

    def test_critical_vitals_call_caregiver_immediately(self):
        snap = _whoop(recovery=15, hrv=30, hrv_baseline=65, rhr=80, rhr_baseline=55)
        mocks = self._run(snap)
        mocks["debate"].assert_not_called()
        mocks["call_caregiver"].assert_called_once()
        data_arg = mocks["call_caregiver"].call_args[0][0]
        self.assertGreater(data_arg["anomaly_level"], 0)

    def test_debate_receives_interval_arg(self):
        snap = _whoop(recovery=30, hrv=42, hrv_baseline=65, rhr=63, rhr_baseline=55)
        mocks = self._run(snap)
        mocks["debate"].assert_called_once()
        _, interval_arg = mocks["debate"].call_args[0]
        self.assertGreater(interval_arg, 0)

    def test_missing_profile_exits_cleanly(self):
        with patch.object(heartbeat, "get_patient_profile", return_value={}):
            heartbeat.run("GHOST-999", max_ticks=5)  # must not raise

    def test_missing_anchor_exits_cleanly(self):
        with (
            patch.object(heartbeat, "get_patient_profile", return_value=_profile()),
            patch.object(heartbeat, "get_anchor",          return_value=None),
        ):
            heartbeat.run("TEST-001", max_ticks=5)

    def test_missing_whoop_row_exits_cleanly(self):
        with (
            patch.object(heartbeat, "get_patient_profile", return_value=_profile()),
            patch.object(heartbeat, "get_anchor",          return_value=_ANCHOR),
            patch.object(heartbeat, "fetch_by_timestamp",  return_value={}),
        ):
            heartbeat.run("TEST-001", max_ticks=5)

    def test_max_ticks_respected(self):
        """Two consecutive normal ticks should both complete without debate."""
        snap = _whoop(recovery=80, hrv=65, hrv_baseline=65, rhr=55, rhr_baseline=55)
        debate_mock = MagicMock()
        import contextlib
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(heartbeat, "get_patient_profile", return_value=_profile()))
            stack.enter_context(patch.object(heartbeat, "get_anchor",          return_value=_ANCHOR))
            stack.enter_context(patch.object(heartbeat, "fetch_by_timestamp",  return_value=snap))
            stack.enter_context(patch.object(heartbeat, "_fire",               side_effect=lambda fn, *a: fn(*a)))
            stack.enter_context(patch.object(heartbeat, "_trigger_debate",     debate_mock))
            stack.enter_context(patch.object(heartbeat.time, "sleep",          return_value=None))
            heartbeat.run("TEST-001", max_ticks=2)
        debate_mock.assert_not_called()


# ── 4. _trigger_debate action routing ────────────────────────────────────────

class TestDebateRouting(TestCase):
    """
    Verify that _trigger_debate starts the same async SSE-backed debate
    that the frontend observes.
    """

    def test_starts_streaming_debate_for_patient(self):
        response = MagicMock()
        response.json.return_value = {"status": "debate started", "stream_url": "/stream/TEST-001"}
        response.raise_for_status.return_value = None
        post_mock = MagicMock(return_value=response)
        data = _whoop(recovery=15, rhr=80, rhr_baseline=55)
        data["anomaly_level"] = 4

        with (
            patch.object(heartbeat, "get_patient_profile", return_value=_profile()),
            patch("heartbeat.requests.post", post_mock),
        ):
            heartbeat._trigger_debate(data, interval=30)

        post_mock.assert_called_once()
        url, = post_mock.call_args.args
        payload = post_mock.call_args.kwargs["json"]
        self.assertEqual(url, f"{heartbeat.DEBATE_BASE_URL}/start-debate/TEST-001")
        self.assertEqual(payload["patientId"], "TEST-001")
        self.assertEqual(payload["interval"], 30)


# ── 5. anomaly level → debate escalation integration ─────────────────────────

class TestAnomalyLevelScenarios(TestCase):
    """
    End-to-end table of WHOOP snapshots → expected anomaly level range.
    Validates the fallback rule-based scorer against realistic patient data.
    """

    CASES = [
        # label,               snap kwargs,                                       min_level, max_level
        ("post_discharge_ok",  dict(recovery=72, hrv=58, hrv_baseline=60,
                                    rhr=62, rhr_baseline=60),                     0, 1),
        ("mild_fatigue",       dict(recovery=44, hrv=50, hrv_baseline=65,
                                    rhr=64, rhr_baseline=58, sleep_perf=60),      1, 2),
        ("moderate_concern",   dict(recovery=28, hrv=38, hrv_baseline=65,
                                    rhr=67, rhr_baseline=55, skin_temp_dev=0.9),  2, 3),
        ("high_concern",       dict(recovery=22, hrv=25, hrv_baseline=65,
                                    rhr=72, rhr_baseline=55, sleep_perf=45),      3, 4),
        ("critical_post_mi",   dict(recovery=12, hrv=20, hrv_baseline=65,
                                    rhr=82, rhr_baseline=55),                     4, 4),
        # Synthetic seed patients
        ("PAT-SYN-L4-002_current",  dict(recovery=16, hrv=28, hrv_baseline=54,
                                          rhr=99, rhr_baseline=74,
                                          skin_temp_dev=1.4, sleep_perf=48),      4, 4),
        ("PAT-SYN-L4-002_history",  dict(recovery=68, hrv=52, hrv_baseline=54,
                                          rhr=76, rhr_baseline=74),               0, 1),
        ("PAT-SYN-L3-001_current",  dict(recovery=29, hrv=37, hrv_baseline=58,
                                          rhr=90, rhr_baseline=74,
                                          skin_temp_dev=0.9, sleep_perf=56),      2, 3),
        ("PAT-SYN-L3-001_history",  dict(recovery=71, hrv=56, hrv_baseline=58,
                                          rhr=75, rhr_baseline=74),               0, 1),
    ]

    def test_scenarios(self):
        for label, kwargs, min_lvl, max_lvl in self.CASES:
            with self.subTest(scenario=label):
                level = _fallback_rule_based_level(_whoop(**kwargs))
                self.assertGreaterEqual(level, min_lvl,
                    msg=f"{label}: expected ≥{min_lvl}, got {level}")
                self.assertLessEqual(level, max_lvl,
                    msg=f"{label}: expected ≤{max_lvl}, got {level}")


if __name__ == "__main__":
    main(verbosity=2)
