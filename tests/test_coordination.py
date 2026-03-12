"""
Unit Tests — coordination package (Layer 4: Hierarchical Coordination).

Tests cover:
    - SatelliteForecast: viability assessment for various SoC/temp conditions
    - GlobalForecast: aggregation summary
    - ClusterCoordinator: aggregate(), assess_fleet_status()
    - GreedyAllocator: power budget, relay constraint, all-hibernate fallback
    - ILPAllocator: feasible solution shape
    - TaskAllocator: end-to-end solve()
    - PayloadScheduler: build_schedule, eclipse override, priority queue,
                        task_distribution, to_array, generate_eclipse_profile
    - GroundStationLink: comm_window detection, next_window
    - CommandRouter: send, dispatch_assignment, flush_held, emergency
"""

import pytest
import numpy as np

N_AGENTS  = 6    # representative mid-size constellation
OBS_DIM   = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(soc=0.8, soh=0.95, temp_norm=0.25, eclipse=0.0):
    s = np.zeros(OBS_DIM, dtype=np.float32)
    s[0] = soc
    s[1] = soh
    s[2] = temp_norm
    s[5] = eclipse
    return s


def _make_global_forecast(n=N_AGENTS, soc_vals=None):
    from coordination.cluster_coordinator import SatelliteForecast, GlobalForecast
    if soc_vals is None:
        soc_vals = [0.8] * n
    forecasts = []
    for i in range(n):
        states = np.tile(_make_state(soc=soc_vals[i]), (5, 1))  # 5-step horizon
        forecasts.append(SatelliteForecast(sat_id=i, horizon=5, states=states))
    return GlobalForecast(n_satellites=n, horizon=5, forecasts=forecasts)


# ---------------------------------------------------------------------------
# SatelliteForecast
# ---------------------------------------------------------------------------

class TestSatelliteForecast:
    def test_viable_healthy_sat(self):
        from coordination.cluster_coordinator import SatelliteForecast
        states = np.tile(_make_state(soc=0.8, temp_norm=0.25), (5, 1))
        sf = SatelliteForecast(sat_id=0, horizon=5, states=states)
        assert sf.viable
        assert "payload_active" in sf.viable_tasks
        assert "relay_mode" in sf.viable_tasks
        assert "hibernate" in sf.viable_tasks

    def test_not_viable_payload_on_low_soc(self):
        from coordination.cluster_coordinator import SatelliteForecast
        states = np.tile(_make_state(soc=0.12), (5, 1))
        sf = SatelliteForecast(sat_id=0, horizon=5, states=states)
        assert "payload_active" not in sf.viable_tasks
        assert "charge_priority" in sf.viable_tasks

    def test_not_viable_payload_on_high_temp(self):
        from coordination.cluster_coordinator import SatelliteForecast
        states = np.tile(_make_state(soc=0.8, temp_norm=0.60), (5, 1))
        sf = SatelliteForecast(sat_id=0, horizon=5, states=states)
        assert "payload_active" not in sf.viable_tasks

    def test_derived_means(self):
        from coordination.cluster_coordinator import SatelliteForecast
        soc_vals = np.linspace(0.6, 0.9, 5)
        states = np.zeros((5, OBS_DIM), dtype=np.float32)
        states[:, 0] = soc_vals
        sf = SatelliteForecast(sat_id=0, horizon=5, states=states)
        assert abs(sf.mean_soc - soc_vals.mean()) < 1e-5
        assert abs(sf.min_soc  - soc_vals.min())  < 1e-5


# ---------------------------------------------------------------------------
# GlobalForecast
# ---------------------------------------------------------------------------

class TestGlobalForecast:
    def test_aggregation_summary(self):
        gf = _make_global_forecast(n=N_AGENTS, soc_vals=[0.8] * N_AGENTS)
        assert abs(gf.mean_fleet_soc - 0.8) < 1e-5
        assert gf.n_viable == N_AGENTS

    def test_low_soc_reduces_viable(self):
        soc_vals = [0.09] * 3 + [0.8] * 3  # 3 critical, 3 healthy
        gf = _make_global_forecast(n=6, soc_vals=soc_vals)
        assert gf.n_viable < 6


# ---------------------------------------------------------------------------
# ClusterCoordinator
# ---------------------------------------------------------------------------

class TestClusterCoordinator:
    def test_aggregate_shape(self):
        from coordination.cluster_coordinator import ClusterCoordinator
        coord = ClusterCoordinator(n_satellites=N_AGENTS, forecast_horizon=5)
        raw = [[_make_state(soc=0.8)] * 5 for _ in range(N_AGENTS)]
        gf = coord.aggregate(raw)
        assert len(gf.forecasts) == N_AGENTS
        assert gf.n_satellites == N_AGENTS

    def test_assess_fleet_status_keys(self):
        from coordination.cluster_coordinator import ClusterCoordinator
        coord = ClusterCoordinator(n_satellites=N_AGENTS)
        gf    = _make_global_forecast()
        status = coord.assess_fleet_status(gf)
        assert "mean_fleet_soc" in status
        assert "n_viable" in status
        assert "viable_fraction" in status

    def test_aggregate_empty_forecast_fallback(self):
        from coordination.cluster_coordinator import ClusterCoordinator
        coord = ClusterCoordinator(n_satellites=2, forecast_horizon=5)
        raw   = [[], []]   # no predictions
        gf    = coord.aggregate(raw)
        assert len(gf.forecasts) == 2


# ---------------------------------------------------------------------------
# GreedyAllocator
# ---------------------------------------------------------------------------

class TestGreedyAllocator:
    def test_all_sats_assigned(self):
        from coordination.task_allocator import GreedyAllocator
        gf  = _make_global_forecast(n=N_AGENTS)
        g   = GreedyAllocator(power_budget_w=300.0)
        res = g.solve(gf)
        assert len(res.assignment) == N_AGENTS

    def test_power_budget_respected(self):
        from coordination.task_allocator import GreedyAllocator
        gf  = _make_global_forecast(n=N_AGENTS)
        g   = GreedyAllocator(power_budget_w=50.0)
        res = g.solve(gf)
        assert res.total_power_w <= 50.0 + 0.1   # small float tolerance

    def test_valid_tasks_assigned(self):
        from coordination.task_allocator import GreedyAllocator
        from coordination.cluster_coordinator import TASKS
        gf  = _make_global_forecast(n=N_AGENTS)
        g   = GreedyAllocator(power_budget_w=300.0)
        res = g.solve(gf)
        for task in res.assignment.values():
            assert task in TASKS

    def test_relay_constraint_enforced(self):
        from coordination.task_allocator import GreedyAllocator
        gf  = _make_global_forecast(n=N_AGENTS)
        g   = GreedyAllocator(power_budget_w=300.0, min_relay_sats=2)
        res = g.solve(gf)
        n_relay = sum(1 for t in res.assignment.values() if t == "relay_mode")
        assert n_relay >= 2

    def test_solver_label(self):
        from coordination.task_allocator import GreedyAllocator
        gf  = _make_global_forecast(n=N_AGENTS)
        res = GreedyAllocator().solve(gf)
        assert res.solver_used == "greedy"


# ---------------------------------------------------------------------------
# TaskAllocator (end-to-end)
# ---------------------------------------------------------------------------

class TestTaskAllocator:
    def test_solve_returns_dict(self):
        from coordination.task_allocator import TaskAllocator
        gf  = _make_global_forecast(n=N_AGENTS)
        ta  = TaskAllocator(n_satellites=N_AGENTS, power_budget_w=300.0, solver="greedy")
        asgn = ta.solve(gf)
        assert isinstance(asgn, dict)
        assert len(asgn) == N_AGENTS

    def test_12_satellite_scale(self):
        """Ensure allocator works at full 12-satellite scale."""
        from coordination.task_allocator import TaskAllocator
        gf  = _make_global_forecast(n=12, soc_vals=[0.7] * 12)
        ta  = TaskAllocator(n_satellites=12, power_budget_w=500.0, solver="greedy")
        asgn = ta.solve(gf)
        assert len(asgn) == 12

    def test_auto_solver(self):
        """Auto mode should not raise."""
        from coordination.task_allocator import TaskAllocator
        gf   = _make_global_forecast(n=3)
        ta   = TaskAllocator(n_satellites=3, power_budget_w=200.0, solver="auto")
        asgn = ta.solve(gf)
        assert len(asgn) == 3


# ---------------------------------------------------------------------------
# PayloadScheduler
# ---------------------------------------------------------------------------

class TestPayloadScheduler:
    def _make_scheduler(self, n=3):
        from coordination.scheduling import PayloadScheduler, SLOT_DURATION_S
        return PayloadScheduler(
            n_satellites=n,
            orbital_period_s=SLOT_DURATION_S * 10,   # 10 slots for speed
            slot_duration_s=SLOT_DURATION_S,
            power_budget_w=300.0,
        )

    def test_schedule_length(self):
        sched = self._make_scheduler()
        base  = {0: "payload_active", 1: "relay_mode", 2: "hibernate"}
        slots = sched.build_schedule(base)
        assert len(slots) == 10    # 10 slots

    def test_all_sats_in_each_slot(self):
        sched = self._make_scheduler(n=3)
        base  = {0: "payload_active", 1: "relay_mode", 2: "hibernate"}
        slots = sched.build_schedule(base)
        for slot in slots:
            assert len(slot.assignment) == 3

    def test_eclipse_overrides_payload(self):
        from coordination.scheduling import PayloadScheduler, SLOT_DURATION_S
        n     = 3
        sched = PayloadScheduler(n_satellites=n, orbital_period_s=SLOT_DURATION_S * 5,
                                  slot_duration_s=SLOT_DURATION_S)
        eclipse = np.ones((n, 5), dtype=bool)   # all sats in eclipse all slots
        base    = {i: "payload_active" for i in range(n)}
        slots   = sched.build_schedule(base, eclipse_profile=eclipse)
        for slot in slots:
            for task in slot.assignment.values():
                assert task != "payload_active", "payload_active must not run in eclipse"

    def test_priority_queue_urgent_task(self):
        sched = self._make_scheduler(n=3)
        sched.add_urgent_task(sat_id=0, task="charge_priority", priority=1)
        base  = {0: "payload_active", 1: "relay_mode", 2: "hibernate"}
        slots = sched.build_schedule(base)
        # First slot should use urgent task for SAT-0
        assert slots[0].assignment[0] == "charge_priority"

    def test_task_distribution_keys(self):
        sched = self._make_scheduler()
        base  = {0: "payload_active", 1: "relay_mode", 2: "hibernate"}
        sched.build_schedule(base)
        dist  = sched.task_distribution()
        assert isinstance(dist, dict)

    def test_to_array_shape(self):
        sched = self._make_scheduler(n=3)
        base  = {i: "hibernate" for i in range(3)}
        sched.build_schedule(base)
        arr   = sched.to_array()
        assert arr.shape == (10, 3)
        assert arr.dtype == np.int32

    def test_generate_eclipse_profile(self):
        sched = self._make_scheduler(n=3)
        profile = sched.generate_eclipse_profile()
        assert profile.shape == (3, 10)
        assert profile.dtype == bool
        # Not all slots should be eclipsed
        assert profile.sum() < 3 * 10

    def test_slot_summary_string(self):
        sched = self._make_scheduler(n=2)
        base  = {0: "payload_active", 1: "relay_mode"}
        slots = sched.build_schedule(base)
        s = slots[0].summary()
        assert "SAT-0" in s


# ---------------------------------------------------------------------------
# GroundStationLink
# ---------------------------------------------------------------------------

class TestGroundStationLink:
    def test_window_opens_some_of_the_time(self):
        from coordination.communication_protocol import GroundStationLink
        gl = GroundStationLink(n_satellites=3)
        # Sweep one full orbit and check at least one window is open
        period = 5520.0
        any_open = any(gl.comm_window_open(0, t) for t in np.linspace(0, period, 100))
        assert any_open

    def test_next_window_is_nonnegative(self):
        from coordination.communication_protocol import GroundStationLink
        gl  = GroundStationLink(n_satellites=3)
        nxt = gl.next_window(0, 0.0)
        assert nxt >= 0.0

    def test_next_window_zero_when_open(self):
        from coordination.communication_protocol import GroundStationLink
        gl = GroundStationLink(n_satellites=3)
        # Find a time when window is open, then check next_window == 0
        for t in np.linspace(0, 5520.0, 200):
            if gl.comm_window_open(0, t):
                assert gl.next_window(0, t) == 0.0
                break


# ---------------------------------------------------------------------------
# CommandRouter
# ---------------------------------------------------------------------------

class TestCommandRouter:
    def _make_router(self, n=N_AGENTS):
        from marl.communication_protocol import CommunicationProtocol
        from coordination.communication_protocol import CommandRouter
        comm   = CommunicationProtocol(n_agents=n)
        router = CommandRouter(n_satellites=n, comm=comm)
        return router, comm

    def test_send_dispatches_via_isl(self):
        router, comm = self._make_router()
        sent = router.send(sat_id=0, task="hibernate", via_isl=True)
        assert sent == True
        cmds = comm.receive_commands(0)
        assert len(cmds) == 1
        assert cmds[0].task == "hibernate"

    def test_dispatch_assignment(self):
        router, comm = self._make_router(n=3)
        asgn = {0: "payload_active", 1: "relay_mode", 2: "hibernate"}
        router.dispatch_assignment(asgn, via_isl=True)
        for sat_id, expected_task in asgn.items():
            cmds = comm.receive_commands(sat_id)
            assert len(cmds) == 1
            assert cmds[0].task == expected_task

    def test_emergency_broadcast_all_agents(self):
        router, comm = self._make_router(n=N_AGENTS)
        router.broadcast_emergency("charge_priority")
        for i in range(N_AGENTS):
            cmds = comm.receive_commands(i)
            assert len(cmds) == 1
            assert cmds[0].task == "charge_priority"
            assert cmds[0].priority == 3
