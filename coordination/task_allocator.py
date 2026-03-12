"""
Task Allocator — Layer 4: Hierarchical Coordination.

Solves the constrained satellite task assignment problem.

Objective:
    maximize  sum_i  mission_value(task_i) * x_i
    subject to:
        sum_i  power(task_i) * x_i  <=  power_budget
        sum_i  [task_i == relay]    >=  min_relay_sats
        SoC_i  > SoC_min if task_i == payload_active
        temp_i < T_max  if task_i == payload_active
        x_i in {0, 1} (one task per satellite)

Two solvers are provided:
    1. GreedyAllocator   — O(n log n) greedy by value/power ratio (reliable)
    2. ILPAllocator      — exact integer linear programming via scipy.optimize
                           (used when cvxpy/ortools is unavailable)

The main TaskAllocator class tries ILP first and falls back to greedy.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from loguru import logger

from coordination.cluster_coordinator import (
    GlobalForecast,
    SatelliteForecast,
    TASKS,
    TASK_POWER_W,
    TASK_VALUE,
)


@dataclass
class AllocationResult:
    """Result of a single task allocation solve."""
    assignment:     Dict[int, str]    # {sat_id: task_name}
    total_value:    float
    total_power_w:  float
    feasible:       bool
    solver_used:    str


# ---------------------------------------------------------------------------
# Greedy allocator
# ---------------------------------------------------------------------------

class GreedyAllocator:
    """
    Greedy task allocator sorted by mission value / power ratio.

    Steps:
        1. For each satellite, compute candidate tasks from viable_tasks.
        2. Score each (satellite, task) pair: value / power.
        3. Greedily assign highest-scoring feasible pair until budget exhausted.
        4. Ensure min_relay_sats constraint is met by post-processing.

    Args:
        power_budget_w (float): total constellation power budget.
        min_relay_sats (int)  : minimum number of relay satellites.
    """

    def __init__(self, power_budget_w: float = 200.0, min_relay_sats: int = 1) -> None:
        self.power_budget_w = power_budget_w
        self.min_relay_sats = min_relay_sats

    def solve(self, global_forecast: GlobalForecast) -> AllocationResult:
        forecasts     = global_forecast.forecasts
        assignment    = {}
        remaining_pwr = self.power_budget_w
        n_relaying    = 0

        # Build (score, sat_id, task) candidate list
        candidates: List[Tuple[float, int, str]] = []
        for sf in forecasts:
            for task in sf.viable_tasks:
                pwr   = TASK_POWER_W[task]
                val   = TASK_VALUE[task]
                score = val / max(pwr, 1.0)
                candidates.append((score, sf.sat_id, task))

        # Sort descending by score
        candidates.sort(key=lambda x: -x[0])

        assigned_sats: set = set()
        for score, sat_id, task in candidates:
            if sat_id in assigned_sats:
                continue
            pwr = TASK_POWER_W[task]
            if pwr <= remaining_pwr:
                assignment[sat_id]  = task
                remaining_pwr      -= pwr
                assigned_sats.add(sat_id)
                if task == "relay_mode":
                    n_relaying += 1

        # Assign hibernate to any unassigned satellite
        for sf in forecasts:
            if sf.sat_id not in assignment:
                assignment[sf.sat_id] = "hibernate"

        # Post-process: enforce minimum relay constraint
        assignment = self._enforce_relay_constraint(assignment, forecasts, n_relaying)

        total_val = sum(TASK_VALUE[t] for t in assignment.values())
        total_pwr = sum(TASK_POWER_W[t] for t in assignment.values())

        return AllocationResult(
            assignment=assignment,
            total_value=total_val,
            total_power_w=total_pwr,
            feasible=total_pwr <= self.power_budget_w,
            solver_used="greedy",
        )

    def _enforce_relay_constraint(
        self,
        assignment: Dict[int, str],
        forecasts: List[SatelliteForecast],
        n_relaying: int,
    ) -> Dict[int, str]:
        """Upgrade a hibernating satellite to relay if constraint is violated."""
        if n_relaying >= self.min_relay_sats:
            return assignment

        for sf in forecasts:
            if assignment.get(sf.sat_id) == "hibernate" and "relay_mode" in sf.viable_tasks:
                assignment[sf.sat_id] = "relay_mode"
                n_relaying += 1
                logger.debug(f"[Alloc] Relay constraint: SAT-{sf.sat_id} upgraded to relay_mode")
            if n_relaying >= self.min_relay_sats:
                break

        return assignment


# ---------------------------------------------------------------------------
# ILP allocator (scipy linprog-based)
# ---------------------------------------------------------------------------

class ILPAllocator:
    """
    Integer Linear Programming task allocator using scipy.optimize.milp.

    Falls back to greedy if scipy MILP is unavailable or infeasible.

    Formulation:
        x[i, j] in {0, 1} — satellite i assigned task j
        Constraints:
            sum_j x[i, j] == 1               for all i  (one task per sat)
            sum_i sum_j power[j]*x[i,j] <= P_budget    (power budget)
            sum_i x[i, relay] >= min_relay              (min relay)
            x[i, payload] == 0 if not viable_payload_i  (feasibility)
        Objective:
            maximize sum_i sum_j value[j] * x[i, j]
    """

    def __init__(self, power_budget_w: float = 200.0, min_relay_sats: int = 1) -> None:
        self.power_budget_w = power_budget_w
        self.min_relay_sats = min_relay_sats
        self._greedy = GreedyAllocator(power_budget_w, min_relay_sats)

    def solve(self, global_forecast: GlobalForecast) -> AllocationResult:
        try:
            return self._solve_milp(global_forecast)
        except Exception as e:
            logger.warning(f"[ILP] MILP failed ({e}), falling back to greedy.")
            result = self._greedy.solve(global_forecast)
            result.solver_used = "greedy_fallback"
            return result

    def _solve_milp(self, global_forecast: GlobalForecast) -> AllocationResult:
        from scipy.optimize import milp, LinearConstraint, Bounds

        forecasts = global_forecast.forecasts
        n         = len(forecasts)
        task_list = TASKS
        m         = len(task_list)
        # Decision variable vector: x[i*m + j] for satellite i, task j

        # Objective: maximize value -> minimize negative value
        c = np.array([-TASK_VALUE[task_list[j]] for j in range(m)] * n, dtype=float)

        # Constraint 1: each satellite gets exactly one task
        A_eq_rows = []
        for i in range(n):
            row = np.zeros(n * m)
            row[i * m: (i + 1) * m] = 1.0
            A_eq_rows.append(row)

        # Constraint 2: total power <= budget
        power_row = np.array([TASK_POWER_W[task_list[j]] for j in range(m)] * n, dtype=float)

        # Constraint 3: relay >= min_relay
        relay_idx = task_list.index("relay_mode")
        relay_row = np.zeros(n * m)
        for i in range(n):
            relay_row[i * m + relay_idx] = 1.0

        # Feasibility: zero out non-viable task columns per satellite
        integrality = np.ones(n * m, dtype=int)
        lb          = np.zeros(n * m)
        ub          = np.ones(n * m)
        for i, sf in enumerate(forecasts):
            for j, task in enumerate(task_list):
                if task not in sf.viable_tasks:
                    ub[i * m + j] = 0.0

        constraints = [
            LinearConstraint(np.array(A_eq_rows), lb=np.ones(n), ub=np.ones(n)),   # exactly 1 task
            LinearConstraint(power_row.reshape(1, -1), lb=-np.inf, ub=np.array([self.power_budget_w])),
            LinearConstraint(relay_row.reshape(1, -1), lb=np.array([float(self.min_relay_sats)]), ub=np.inf),
        ]
        bounds = Bounds(lb=lb, ub=ub)

        result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)

        if not result.success:
            raise RuntimeError(f"MILP infeasible: {result.message}")

        x = np.round(result.x).astype(int)
        assignment = {}
        for i in range(n):
            for j in range(m):
                if x[i * m + j] == 1:
                    assignment[forecasts[i].sat_id] = task_list[j]
                    break
            else:
                assignment[forecasts[i].sat_id] = "hibernate"

        total_val = sum(TASK_VALUE[t] for t in assignment.values())
        total_pwr = sum(TASK_POWER_W[t] for t in assignment.values())

        return AllocationResult(
            assignment=assignment,
            total_value=total_val,
            total_power_w=total_pwr,
            feasible=result.success,
            solver_used="scipy_milp",
        )


# ---------------------------------------------------------------------------
# Main TaskAllocator (tries ILP first, falls back to greedy)
# ---------------------------------------------------------------------------

class TaskAllocator:
    """
    Main entry point for constrained satellite task assignment.

    Tries scipy MILP first (exact, optimal). Falls back to greedy on failure.

    Args:
        n_satellites   (int)  : number of satellites (supports up to 12)
        power_budget_w (float): constellation-level power budget in Watts
        min_relay_sats (int)  : minimum relay satellites (for comm coverage)
        solver         (str)  : "auto" | "greedy" | "ilp"
    """

    def __init__(
        self,
        n_satellites:   int,
        power_budget_w: float = 200.0,
        min_relay_sats: int   = 1,
        solver:         str   = "auto",
    ) -> None:
        self.n_satellites   = n_satellites
        self.power_budget_w = power_budget_w
        self.min_relay_sats = min_relay_sats
        self.solver         = solver

        self._greedy = GreedyAllocator(power_budget_w, min_relay_sats)
        self._ilp    = ILPAllocator(power_budget_w, min_relay_sats)

    def solve(self, global_forecast: GlobalForecast) -> Dict[int, str]:
        """
        Solve the task allocation and return an assignment dict.

        Args:
            global_forecast: aggregated constellation forecast

        Returns:
            {satellite_id: task_name}
        """
        if self.solver == "greedy":
            result = self._greedy.solve(global_forecast)
        elif self.solver == "ilp":
            result = self._ilp.solve(global_forecast)
        else:
            # Auto: try ILP, fallback greedy
            result = self._ilp.solve(global_forecast)

        logger.info(
            f"[TaskAlloc] solver={result.solver_used} | "
            f"value={result.total_value:.2f} | "
            f"power={result.total_power_w:.1f}W / {self.power_budget_w:.0f}W | "
            f"feasible={result.feasible}"
        )
        return result.assignment
