"""
Layer 4 — coordination package (Hierarchical Coordination Layer).

Exports:
    SatelliteForecast       — per-satellite k-step world model forecast summary
    GlobalForecast          — aggregated constellation-level forecast
    ClusterCoordinator      — main Algorithm 5 coordinator
    AllocationResult        — task allocation output dataclass
    GreedyAllocator         — O(n log n) greedy task solver
    ILPAllocator            — scipy MILP exact task solver
    TaskAllocator           — auto solver (ILP + greedy fallback)
    ScheduleSlot            — one time-slot in the orbital schedule
    PayloadScheduler        — builds full orbital-period schedule
    GroundStation           — ground station dataclass
    GroundStationLink       — models satellite-to-ground comm windows
    CommandRouter           — priority command routing (coordinator -> agents)
    CommunicationProtocol   — re-export from marl (L3 peer comms)
    CommandMessage          — re-export from marl
"""

from coordination.cluster_coordinator import (
    SatelliteForecast,
    GlobalForecast,
    ClusterCoordinator,
    TASKS,
    TASK_POWER_W,
    TASK_VALUE,
)
from coordination.task_allocator import (
    AllocationResult,
    GreedyAllocator,
    ILPAllocator,
    TaskAllocator,
)
from coordination.scheduling import (
    ScheduleSlot,
    PriorityTask,
    PayloadScheduler,
    LEO_PERIOD_S,
    SLOT_DURATION_S,
)
from coordination.communication_protocol import (
    GroundStation,
    GroundStationLink,
    CommandRouter,
    DEFAULT_GROUND_STATIONS,
    CommunicationProtocol,   # re-export from marl
    CommandMessage,          # re-export from marl
)

__all__ = [
    # Cluster coordinator
    "SatelliteForecast",
    "GlobalForecast",
    "ClusterCoordinator",
    "TASKS",
    "TASK_POWER_W",
    "TASK_VALUE",
    # Task allocator
    "AllocationResult",
    "GreedyAllocator",
    "ILPAllocator",
    "TaskAllocator",
    # Scheduler
    "ScheduleSlot",
    "PriorityTask",
    "PayloadScheduler",
    "LEO_PERIOD_S",
    "SLOT_DURATION_S",
    # Communication
    "GroundStation",
    "GroundStationLink",
    "CommandRouter",
    "DEFAULT_GROUND_STATIONS",
    "CommunicationProtocol",
    "CommandMessage",
]
