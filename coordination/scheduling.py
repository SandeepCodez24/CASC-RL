"""
Payload Scheduler — Layer 4: Hierarchical Coordination.

Builds a temporal task schedule over one full orbital period (~92 minutes
for a 400 km LEO orbit). Manages:
    - Time-slotted mission plan per satellite
    - Communication window detection (pass-over ground stations)
    - ISL (Inter-Satellite Link) availability windows
    - Priority queue for urgent tasks (emergency relay > science payload)
    - Schedule conflict resolution (two sats cannot use same comm slot)

Schedule output example (for 3 satellites, 1 orbit):
    t=00:00  SAT-0: payload_active  SAT-1: relay_mode  SAT-2: hibernate
    t=05:00  SAT-0: relay_mode      SAT-1: payload_active  SAT-2: charge_priority
    ...

The schedule is produced as a list of ScheduleSlot objects, one per time window.
"""

from __future__ import annotations

import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from loguru import logger

from coordination.cluster_coordinator import TASKS, TASK_POWER_W, TASK_VALUE


# Orbital period for 400 km LEO (seconds)
LEO_PERIOD_S = 92 * 60      # 5520 seconds

# Duration of one scheduling slot (seconds)
SLOT_DURATION_S = 5 * 60    # 5 minutes


@dataclass(order=True)
class PriorityTask:
    """Task with priority for the priority queue (lower value = higher priority)."""
    priority:   int
    task:       str = field(compare=False)
    sat_id:     int = field(compare=False)
    start_time: float = field(compare=False)


@dataclass
class ScheduleSlot:
    """
    One time slot in the constellation schedule.

    Args:
        slot_idx      : index of this slot in the schedule
        start_time_s  : seconds from epoch
        end_time_s    : start_time_s + SLOT_DURATION_S
        assignment    : {sat_id: task_name} for this slot
        eclipse_flags : {sat_id: bool} — True if satellite is in eclipse
        isl_active    : True if ISL links are available this slot
    """
    slot_idx:     int
    start_time_s: float
    end_time_s:   float
    assignment:   Dict[int, str]
    eclipse_flags: Dict[int, bool] = field(default_factory=dict)
    isl_active:   bool = True

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    def summary(self) -> str:
        t_min = int(self.start_time_s // 60)
        parts = [f"t={t_min:03d}min"]
        parts += [f"SAT-{k}:{v}" for k, v in sorted(self.assignment.items())]
        return "  ".join(parts)


class PayloadScheduler:
    """
    Temporal mission scheduler for one orbital period.

    Divides the orbital period into fixed-size slots and allocates tasks to
    each satellite per slot, respecting:
        - Eclipse periods (no payload_active during eclipse)
        - Power budgets per slot
        - Priority tasks (via a min-heap priority queue)
        - ISL link availability windows

    Args:
        n_satellites     (int): number of satellites
        orbital_period_s (float): orbital period in seconds. Default 5520s (92 min).
        slot_duration_s  (float): duration of each schedule slot. Default 300s (5 min).
        power_budget_w   (float): power budget per slot. Default 200W.
    """

    def __init__(
        self,
        n_satellites:     int,
        orbital_period_s: float = LEO_PERIOD_S,
        slot_duration_s:  float = SLOT_DURATION_S,
        power_budget_w:   float = 200.0,
    ) -> None:
        self.n_satellites     = n_satellites
        self.orbital_period_s = orbital_period_s
        self.slot_duration_s  = slot_duration_s
        self.power_budget_w   = power_budget_w

        self.n_slots = int(np.ceil(orbital_period_s / slot_duration_s))
        self._priority_queue: List[PriorityTask] = []   # min-heap
        self._schedule: List[ScheduleSlot] = []

    # ------------------------------------------------------------------
    # Schedule building
    # ------------------------------------------------------------------

    def build_schedule(
        self,
        base_assignment:  Dict[int, str],        # from TaskAllocator
        eclipse_profile:  Optional[np.ndarray] = None,  # (n_sat, n_slots) bool
        isl_profile:      Optional[np.ndarray] = None,  # (n_slots,) bool
    ) -> List[ScheduleSlot]:
        """
        Construct a full orbital period schedule.

        Args:
            base_assignment : default task per satellite from TaskAllocator
            eclipse_profile : boolean eclipse mask (n_sat, n_slots).
                              None: assume no eclipse.
            isl_profile     : boolean ISL availability per slot.
                              None: assume always available.

        Returns:
            List of ScheduleSlot objects (one per time window)
        """
        self._schedule = []
        urgent_tasks   = self._drain_priority_queue()

        for slot_idx in range(self.n_slots):
            t_start = slot_idx * self.slot_duration_s
            t_end   = t_start + self.slot_duration_s

            # Eclipse flags for this slot
            eclipse_flags = {}
            if eclipse_profile is not None:
                for sat_id in range(self.n_satellites):
                    eclipse_flags[sat_id] = bool(eclipse_profile[sat_id, slot_idx])
            else:
                eclipse_flags = {i: False for i in range(self.n_satellites)}

            # ISL availability
            isl_active = bool(isl_profile[slot_idx]) if isl_profile is not None else True

            # Build per-satellite assignment for this slot
            slot_assignment = self._assign_slot(
                base_assignment=base_assignment,
                urgent_tasks=urgent_tasks,
                eclipse_flags=eclipse_flags,
                slot_idx=slot_idx,
            )

            slot = ScheduleSlot(
                slot_idx=slot_idx,
                start_time_s=t_start,
                end_time_s=t_end,
                assignment=slot_assignment,
                eclipse_flags=eclipse_flags,
                isl_active=isl_active,
            )
            self._schedule.append(slot)

        logger.info(
            f"[Scheduler] Built schedule: {len(self._schedule)} slots | "
            f"{self.n_satellites} satellites | period={self.orbital_period_s/60:.1f} min"
        )
        return self._schedule

    def _assign_slot(
        self,
        base_assignment: Dict[int, str],
        urgent_tasks:    Dict[int, str],
        eclipse_flags:   Dict[int, bool],
        slot_idx:        int,
    ) -> Dict[int, str]:
        """Assign tasks for one slot, applying eclipse and urgent overrides."""
        remaining_pwr = self.power_budget_w
        assignment    = {}
        relay_count   = 0

        for sat_id in range(self.n_satellites):
            # Urgent task takes priority
            if sat_id in urgent_tasks:
                task = urgent_tasks[sat_id]
            else:
                task = base_assignment.get(sat_id, "hibernate")

            # Eclipse constraint: cannot run payload during eclipse
            if eclipse_flags.get(sat_id, False) and task == "payload_active":
                task = "charge_priority"   # switch to charging during eclipse

            # Power budget check
            pwr = TASK_POWER_W[task]
            if pwr > remaining_pwr:
                task = "hibernate"   # downgrade to hibernate if over budget
                pwr  = TASK_POWER_W["hibernate"]

            # Relay conflict resolution (slot-level): max 1 per slot is recommended
            if task == "relay_mode":
                relay_count += 1
                if relay_count > max(self.n_satellites // 3, 1):
                    task = "payload_off"

            remaining_pwr -= pwr
            assignment[sat_id] = task

        return assignment

    # ------------------------------------------------------------------
    # Priority queue for urgent tasks
    # ------------------------------------------------------------------

    def add_urgent_task(
        self,
        sat_id:     int,
        task:       str,
        priority:   int = 1,
        start_time: float = 0.0,
    ) -> None:
        """
        Add an urgent task to the priority queue.

        Args:
            sat_id    : target satellite index
            task      : task name (from TASKS)
            priority  : integer, lower = higher priority (1 = urgent, 5 = low)
            start_time: desired start time in seconds
        """
        item = PriorityTask(priority=priority, task=task, sat_id=sat_id, start_time=start_time)
        heapq.heappush(self._priority_queue, item)
        logger.debug(f"[Scheduler] Urgent task queued: SAT-{sat_id} -> {task} (priority={priority})")

    def _drain_priority_queue(self) -> Dict[int, str]:
        """Return one urgent task per satellite (highest priority first)."""
        urgent: Dict[int, str] = {}
        while self._priority_queue:
            item = heapq.heappop(self._priority_queue)
            if item.sat_id not in urgent:
                urgent[item.sat_id] = item.task
        return urgent

    # ------------------------------------------------------------------
    # Schedule analysis utilities
    # ------------------------------------------------------------------

    def get_schedule(self) -> List[ScheduleSlot]:
        """Return the most recently built schedule."""
        return self._schedule

    def task_distribution(self) -> Dict[str, int]:
        """Count total slot-task assignments across the full schedule."""
        counts: Dict[str, int] = {t: 0 for t in TASKS}
        for slot in self._schedule:
            for task in slot.assignment.values():
                counts[task] = counts.get(task, 0) + 1
        return counts

    def print_schedule(self, max_slots: int = 20) -> None:
        """Print a human-readable summary of the first max_slots slots."""
        print(f"\n--- Payload Schedule ({self.n_satellites} satellites, "
              f"{len(self._schedule)} slots) ---")
        for slot in self._schedule[:max_slots]:
            print(f"  {slot.summary()}")
        if len(self._schedule) > max_slots:
            print(f"  ... ({len(self._schedule) - max_slots} more slots)")

    def to_array(self) -> np.ndarray:
        """
        Export schedule as integer array for vectorized processing.

        Returns:
            np.ndarray shape (n_slots, n_satellites), dtype int32
            values are indices into TASKS list.
        """
        task_to_idx = {t: i for i, t in enumerate(TASKS)}
        arr = np.zeros((len(self._schedule), self.n_satellites), dtype=np.int32)
        for s, slot in enumerate(self._schedule):
            for sat_id, task in slot.assignment.items():
                arr[s, sat_id] = task_to_idx.get(task, 0)
        return arr

    def generate_eclipse_profile(
        self,
        orbital_phase_offset: float = 0.0,
    ) -> np.ndarray:
        """
        Generate a simplified eclipse profile for testing.

        Assumes eclipse lasts ~35% of each orbit (typical LEO).
        Each satellite has a phase offset of 360/n_satellites degrees.

        Returns:
            np.ndarray shape (n_satellites, n_slots) bool
        """
        eclipse_fraction = 0.35
        eclipse_slots    = int(self.n_slots * eclipse_fraction)
        profile = np.zeros((self.n_satellites, self.n_slots), dtype=bool)

        for sat_id in range(self.n_satellites):
            offset = int((sat_id / self.n_satellites) * self.n_slots)
            for s in range(eclipse_slots):
                idx = (offset + s) % self.n_slots
                profile[sat_id, idx] = True

        return profile
