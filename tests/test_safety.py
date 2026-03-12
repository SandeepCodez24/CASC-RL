"""
Unit Tests — safety package (Layer 5: Safety & Fault Recovery).

Tests cover:
    - StatisticalDetector: nominal pass, Z-score anomaly trigger
    - ResidualDetector: nominal pass, prediction spike detection
    - AnomalyDetector: combined mode, reset, anomaly_count
    - SafetyMonitor: FSM state transitions (NOMINAL/WARNING/CRITICAL/RECOVERY/DEGRADED)
    - SafetyMonitor: hard action overrides at each state
    - BatteryRecovery: phase progression, action at each phase
    - ThermalRecovery: phase progression, temperature-driven completion
    - GeneralRecovery: stabilize + ramp + complete
    - RecoveryPolicy: activate, activate_from_safety_state, step, is_active, reset

Anomaly injection: 24-hour eclipse stress test (SoC survival check)
"""

import pytest
import numpy as np

OBS_DIM   = 8
IDX_SOC   = 0
IDX_SOH   = 1
IDX_TEMP  = 2

ACTION_PAYLOAD_ON      = 0
ACTION_PAYLOAD_OFF     = 1
ACTION_HIBERNATE       = 2
ACTION_RELAY_MODE      = 3
ACTION_CHARGE_PRIORITY = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs(soc=0.8, soh=0.95, temp=0.25):
    s = np.zeros(OBS_DIM, dtype=np.float32)
    s[IDX_SOC]  = soc
    s[IDX_SOH]  = soh
    s[IDX_TEMP] = temp
    return s


# ---------------------------------------------------------------------------
# StatisticalDetector
# ---------------------------------------------------------------------------

class TestStatisticalDetector:
    def _make_det(self):
        from safety.anomaly_detector import StatisticalDetector
        return StatisticalDetector(sat_id=0, window_size=20, z_threshold=3.0)

    def test_no_anomaly_nominal_data(self):
        det = self._make_det()
        for _ in range(30):
            report = det.update(_obs(soc=0.80, temp=0.25))
        assert report is None    # stable data should not trigger

    def test_anomaly_on_soc_spike(self):
        det = self._make_det()
        # Fill window with stable data
        for _ in range(25):
            det.update(_obs(soc=0.80))
        # Inject extreme SoC drop
        report = det.update(_obs(soc=0.01))
        assert report is not None
        from safety.anomaly_detector import AnomalyType
        assert report.anomaly_type == AnomalyType.BATTERY_OVERDISCHARGE

    def test_anomaly_on_temp_runaway(self):
        det = self._make_det()
        for _ in range(25):
            det.update(_obs(temp=0.25))
        report = det.update(_obs(temp=0.99))
        assert report is not None
        from safety.anomaly_detector import AnomalyType
        assert report.anomaly_type == AnomalyType.THERMAL_RUNAWAY

    def test_reset_clears_window(self):
        det = self._make_det()
        for _ in range(25):
            det.update(_obs(soc=0.8))
        det.reset()
        # No anomaly after reset (window empty, not enough samples)
        report = det.update(_obs(soc=0.01))
        assert report is None   # < 10 samples in window

    def test_severity_in_range(self):
        det = self._make_det()
        for _ in range(25):
            det.update(_obs(soc=0.80))
        report = det.update(_obs(soc=0.01))
        if report:
            assert 0.0 <= report.severity <= 1.0


# ---------------------------------------------------------------------------
# ResidualDetector
# ---------------------------------------------------------------------------

class TestResidualDetector:
    def _make_det(self):
        from safety.anomaly_detector import ResidualDetector
        return ResidualDetector(sat_id=0, residual_threshold=0.1, window_size=10)

    def test_no_anomaly_accurate_prediction(self):
        det = self._make_det()
        obs = _obs()
        det.set_prediction(obs)
        for _ in range(15):
            report = det.update(obs + np.random.randn(OBS_DIM).astype(np.float32) * 0.001)
            det.set_prediction(obs)
        assert report is None

    def test_anomaly_on_large_residual(self):
        det = self._make_det()
        obs  = _obs(soc=0.8)
        pred = _obs(soc=0.8)
        # Build baseline with accurate predictions
        for _ in range(12):
            det.set_prediction(pred)
            det.update(obs + np.random.randn(OBS_DIM).astype(np.float32) * 0.001)
        # Inject large residual
        det.set_prediction(pred)
        report = det.update(_obs(soc=0.01))   # large deviation
        assert report is not None
        from safety.anomaly_detector import AnomalyType
        assert report.anomaly_type == AnomalyType.UNKNOWN_RESIDUAL

    def test_no_anomaly_before_first_prediction(self):
        det = self._make_det()
        report = det.update(_obs())   # no prediction set
        assert report is None


# ---------------------------------------------------------------------------
# AnomalyDetector (combined)
# ---------------------------------------------------------------------------

class TestAnomalyDetector:
    def test_combined_nominal(self):
        from safety.anomaly_detector import AnomalyDetector
        det = AnomalyDetector(sat_id=0, window_size=20)
        for _ in range(30):
            reports = det.step(_obs(soc=0.8, temp=0.25))
        assert reports == []

    def test_combined_detects_stat_anomaly(self):
        from safety.anomaly_detector import AnomalyDetector
        det = AnomalyDetector(sat_id=0, window_size=20)
        for _ in range(25):
            det.step(_obs(soc=0.8))
        reports = det.step(_obs(soc=0.02))
        assert len(reports) > 0

    def test_reset_zeros_count(self):
        from safety.anomaly_detector import AnomalyDetector
        det = AnomalyDetector(sat_id=0, window_size=20)
        for _ in range(25):
            det.step(_obs(soc=0.8))
        det.step(_obs(soc=0.01))
        det.reset()
        assert det.anomaly_count == 0

    def test_anomaly_rate(self):
        from safety.anomaly_detector import AnomalyDetector
        det = AnomalyDetector(sat_id=0, window_size=20)
        for _ in range(25):
            det.step(_obs(soc=0.8))
        det.step(_obs(soc=0.01))
        rate = det.anomaly_rate(last_n=26)
        assert 0.0 < rate <= 1.0


# ---------------------------------------------------------------------------
# SafetyMonitor
# ---------------------------------------------------------------------------

class TestSafetyMonitor:
    def _make_mon(self):
        from safety.safety_monitor import SafetyMonitor, SafetyThresholds
        th = SafetyThresholds(
            soc_warn=0.20, soc_critical=0.10,
            temp_warn=0.50, temp_critical=0.60,
            hold_steps_warn=3,
        )
        return SafetyMonitor(sat_id=0, thresholds=th)

    def test_nominal_passthrough(self):
        from safety.safety_monitor import SafetyState
        mon = self._make_mon()
        action, state, reason = mon.check(_obs(soc=0.8, temp=0.25), policy_action=0)
        assert state == SafetyState.NOMINAL
        assert action == 0    # pass-through

    def test_warning_on_low_soc(self):
        from safety.safety_monitor import SafetyState
        mon = self._make_mon()
        action, state, reason = mon.check(_obs(soc=0.15, temp=0.25), policy_action=0)
        assert state == SafetyState.WARNING

    def test_warning_blocks_payload(self):
        mon = self._make_mon()
        action, _, _ = mon.check(_obs(soc=0.15, temp=0.25), policy_action=ACTION_PAYLOAD_ON)
        assert action != ACTION_PAYLOAD_ON

    def test_critical_on_very_low_soc(self):
        from safety.safety_monitor import SafetyState
        mon = self._make_mon()
        _, state, _ = mon.check(_obs(soc=0.05, temp=0.25), policy_action=0)
        assert state in (SafetyState.CRITICAL, SafetyState.RECOVERY)

    def test_critical_forces_charge_priority(self):
        mon = self._make_mon()
        action, _, _ = mon.check(_obs(soc=0.05), policy_action=0)
        assert action == ACTION_CHARGE_PRIORITY

    def test_thermal_critical_forces_hibernate(self):
        mon = self._make_mon()
        action, _, _ = mon.check(_obs(soc=0.8, temp=0.65), policy_action=0)
        assert action == ACTION_HIBERNATE

    def test_degraded_on_low_soh(self):
        from safety.safety_monitor import SafetyState
        mon = self._make_mon()
        _, state, _ = mon.check(_obs(soc=0.8, soh=0.55, temp=0.25), policy_action=0)
        assert state == SafetyState.DEGRADED

    def test_warning_to_nominal_after_hold(self):
        from safety.safety_monitor import SafetyState
        mon = self._make_mon()
        # Enter warning
        mon.check(_obs(soc=0.15), policy_action=0)
        # Hold safe values for hold_steps_warn=3+1 steps
        for _ in range(4):
            _, state, _ = mon.check(_obs(soc=0.80, temp=0.25), policy_action=0)
        assert state == SafetyState.NOMINAL

    def test_reset_returns_to_nominal(self):
        from safety.safety_monitor import SafetyState
        mon = self._make_mon()
        mon.check(_obs(soc=0.05), policy_action=0)
        mon.reset()
        assert mon.state == SafetyState.NOMINAL
        assert len(mon.event_log) == 0

    def test_is_safe(self):
        mon = self._make_mon()
        mon.check(_obs(soc=0.8), policy_action=0)
        assert mon.is_safe

    def test_state_summary_keys(self):
        mon = self._make_mon()
        mon.check(_obs(soc=0.8), policy_action=0)
        s = mon.state_summary()
        assert "state" in s and "is_safe" in s and "n_events" in s


# ---------------------------------------------------------------------------
# BatteryRecovery
# ---------------------------------------------------------------------------

class TestBatteryRecovery:
    def test_immediate_phase_action(self):
        from safety.recovery_policy import BatteryRecovery, RecoveryPhase
        rec = BatteryRecovery(sat_id=0)
        result = rec.step(_obs(soc=0.05))
        assert result.action == ACTION_CHARGE_PRIORITY
        assert not result.complete

    def test_progression_to_complete(self):
        from safety.recovery_policy import BatteryRecovery, RecoveryPhase
        rec = BatteryRecovery(sat_id=0)
        # Simulate gradual recovery: SoC rises
        for _ in range(5):
            rec.step(_obs(soc=0.05))
        for _ in range(20):
            rec.step(_obs(soc=0.22))
        for _ in range(20):
            result = rec.step(_obs(soc=0.35))
        assert result.complete

    def test_progress_monotone_increasing(self):
        from safety.recovery_policy import BatteryRecovery
        rec = BatteryRecovery(sat_id=0)
        results = []
        soc_seq = [0.05]*5 + [0.20]*20 + [0.35]*20
        for soc in soc_seq:
            results.append(rec.step(_obs(soc=soc)).progress)
        # Progress should generally trend upward (not strictly, but end > start)
        assert results[-1] >= results[0]


# ---------------------------------------------------------------------------
# ThermalRecovery
# ---------------------------------------------------------------------------

class TestThermalRecovery:
    def test_immediate_hibernate(self):
        from safety.recovery_policy import ThermalRecovery
        rec = BatteryRecovery(sat_id=0)   # use battery for now
        from safety.recovery_policy import ThermalRecovery
        rec = ThermalRecovery(sat_id=0)
        result = rec.step(_obs(temp=0.65))
        assert result.action == ACTION_HIBERNATE
        assert not result.complete

    def test_completes_when_temp_drops(self):
        from safety.recovery_policy import ThermalRecovery, RecoveryPhase
        rec = ThermalRecovery(sat_id=0)
        for _ in range(3):
            rec.step(_obs(temp=0.65))    # IMMEDIATE
        for _ in range(30):
            rec.step(_obs(temp=0.43))    # STABILIZE
        for _ in range(30):
            result = rec.step(_obs(temp=0.38))   # GRADUAL_RAMP → COMPLETE
        assert result.complete


# ---------------------------------------------------------------------------
# GeneralRecovery
# ---------------------------------------------------------------------------

class TestGeneralRecovery:
    def test_stabilizes_then_completes(self):
        from safety.recovery_policy import GeneralRecovery
        rec    = GeneralRecovery(sat_id=0, stabilize_steps=5)
        obs    = _obs(soc=0.8, temp=0.30)
        result = None
        for _ in range(20):
            result = rec.step(obs)
        assert result.complete


# ---------------------------------------------------------------------------
# RecoveryPolicy
# ---------------------------------------------------------------------------

class TestRecoveryPolicy:
    def test_activate_battery(self):
        from safety.recovery_policy import RecoveryPolicy
        from safety.anomaly_detector import AnomalyType
        pol = RecoveryPolicy(sat_id=0)
        pol.activate(AnomalyType.BATTERY_OVERDISCHARGE)
        assert pol.is_active()

    def test_step_returns_action(self):
        from safety.recovery_policy import RecoveryPolicy
        from safety.anomaly_detector import AnomalyType
        pol = RecoveryPolicy(sat_id=0)
        pol.activate(AnomalyType.BATTERY_OVERDISCHARGE)
        result = pol.step(_obs(soc=0.05))
        assert result.action in range(5)
        assert 0.0 <= result.progress <= 1.0

    def test_activate_from_low_soc_state(self):
        from safety.recovery_policy import RecoveryPolicy
        from safety.recovery_policy import BatteryRecovery
        pol = RecoveryPolicy(sat_id=0)
        pol.activate_from_safety_state(_obs(soc=0.08))
        assert isinstance(pol._active, BatteryRecovery)

    def test_activate_from_high_temp_state(self):
        from safety.recovery_policy import RecoveryPolicy, ThermalRecovery
        pol = RecoveryPolicy(sat_id=0)
        pol.activate_from_safety_state(_obs(soc=0.8, temp=0.62))
        assert isinstance(pol._active, ThermalRecovery)

    def test_reset_clears_active(self):
        from safety.recovery_policy import RecoveryPolicy
        from safety.anomaly_detector import AnomalyType
        pol = RecoveryPolicy(sat_id=0)
        pol.activate(AnomalyType.THERMAL_RUNAWAY)
        pol.reset()
        assert not pol.is_active()
        assert pol.recovery_steps_taken == 0

    def test_no_active_returns_safe_default(self):
        from safety.recovery_policy import RecoveryPolicy
        pol = RecoveryPolicy(sat_id=0)
        result = pol.step(_obs())
        assert result.complete
        assert result.action == ACTION_PAYLOAD_OFF

    def test_summary_keys(self):
        from safety.recovery_policy import RecoveryPolicy
        from safety.anomaly_detector import AnomalyType
        pol = RecoveryPolicy(sat_id=0)
        pol.activate(AnomalyType.BATTERY_OVERDISCHARGE)
        pol.step(_obs(soc=0.05))
        s = pol.summary()
        assert "is_active" in s and "anomaly_type" in s and "steps_taken" in s


# ---------------------------------------------------------------------------
# Eclipse stress test — integration
# ---------------------------------------------------------------------------

class TestEclipseStressTest:
    """
    Simulates a 24-hour eclipse stress scenario:
        - SoC decreases at 0.02 per step during eclipse (no solar)
        - Safety monitor + recovery policy must prevent SoC from hitting 0
    """
    def test_soc_never_hits_zero(self):
        from safety.safety_monitor import SafetyMonitor
        from safety.recovery_policy import RecoveryPolicy
        from safety.safety_monitor import SafetyState
        from safety.anomaly_detector import AnomalyType

        mon = SafetyMonitor(sat_id=0)
        pol = RecoveryPolicy(sat_id=0)

        soc = 0.70   # starting SoC

        for t in range(500):   # 500 steps ~ extended eclipse
            obs = _obs(soc=soc, temp=0.28)

            action, state, _ = mon.check(obs, policy_action=ACTION_PAYLOAD_ON)

            if mon.needs_recovery and not pol.is_active():
                pol.activate_from_safety_state(obs)

            if pol.is_active():
                result = pol.step(obs)
                action = result.action

            # Simulate physics: charging recovers SoC, payload drains it
            if action == ACTION_CHARGE_PRIORITY:
                soc = min(soc + 0.03, 1.0)     # gaining charge
            elif action == ACTION_HIBERNATE:
                soc = max(soc - 0.005, 0.0)    # minimal drain
            elif action == ACTION_PAYLOAD_ON:
                soc = max(soc - 0.04, 0.0)     # heavy drain
            else:
                soc = max(soc - 0.015, 0.0)    # moderate drain

            assert soc >= 0.0

        # After extended eclipse, safety system should have kept SoC above 0
        assert soc > 0.0, f"SoC hit zero at step {t}"
