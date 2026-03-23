"""
metrics.py — Research-Grade Evaluation Metrics.

All metrics defined in PROJECT_DOCUMENT.md §8 (Phase 8: Evaluation & Benchmarking).
Computes scalar summaries, trajectory statistics, and publication-ready tables
suitable for IEEE/AAAI/NeurIPS paper comparison tables.

Metric categories:
  - Battery health   : lifetime (SoH→70%), SoC min, SoC std, DoD histogram
  - Mission success  : success rate, tasks/orbit, coordination efficiency
  - Safety           : thermal violations, SoC critical events, recovery rate
  - Learning quality : convergence episode, reward stability, sample efficiency
  - Scaling          : per-satellite cost, wall-clock time
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import json
import csv
import os


# ─────────────────────────────────────────────────────────────────────────────
# Episode Result Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    """
    Full record for one evaluation episode across all satellites.
    Populated by the EpisodeRunner in experiment_runner.py.
    """
    episode_id:        int
    n_satellites:      int
    seed:              int
    total_reward:      float

    # Per-step trajectories (list of np.ndarray, shape per step = (n_sat,))
    soc_trajectory:    List[np.ndarray] = field(default_factory=list)   # SoC per sat
    soh_trajectory:    List[np.ndarray] = field(default_factory=list)   # SoH per sat
    temp_trajectory:   List[np.ndarray] = field(default_factory=list)   # Temp (normalized)
    action_trajectory: List[np.ndarray] = field(default_factory=list)   # action per sat
    reward_trajectory: List[float]      = field(default_factory=list)   # sum of rewards

    # Safety events (step indices)
    soc_critical_steps:   List[int] = field(default_factory=list)  # SoC < 0.15
    thermal_viol_steps:   List[int] = field(default_factory=list)  # T_norm > 0.6 (60°C)
    safety_override_steps: List[int] = field(default_factory=list) # safety gate fired

    # Task allocation
    tasks_scheduled:  int = 0
    tasks_completed:  int = 0

    # Metadata
    episode_length:    int   = 0
    algorithm:         str   = "CASC-RL"
    n_satellites_cfg:  int   = 3


# ─────────────────────────────────────────────────────────────────────────────
# Core Metric Computations
# ─────────────────────────────────────────────────────────────────────────────

class MetricComputer:
    """
    Computes all research-grade evaluation metrics from a list of EpisodeResults.

    Usage:
        mc      = MetricComputer(soc_critical_threshold=0.15,
                                 soh_lifetime_threshold=0.70)
        summary = mc.compute_all(results)
        mc.print_table(summary)
        mc.save_json(summary, "results/casc_rl_metrics.json")
    """

    def __init__(
        self,
        soc_critical_threshold: float = 0.15,  # SoC < this → critical event
        soh_lifetime_threshold: float = 0.70,  # SoH drops below → battery end-of-life
        thermal_norm_threshold: float = 0.60,  # normalized temp > this → violation
        n_steps_per_orbit:      int   = 552,   # 552 × 10s = 92 min (LEO period)
    ):
        self.soc_min_thr    = soc_critical_threshold
        self.soh_eol_thr    = soh_lifetime_threshold
        self.thermal_thr    = thermal_norm_threshold
        self.steps_per_orbit = n_steps_per_orbit

    # ── High-Level Entry Point ────────────────────────────────────────────────

    def compute_all(self, results: List[EpisodeResult]) -> Dict[str, Any]:
        """
        Compute full metrics summary over a list of evaluation episodes.

        Returns a dict with keys matching the paper metrics table:
          battery, mission, safety, learning, meta
        """
        if not results:
            raise ValueError("Empty results list — cannot compute metrics.")

        summary: Dict[str, Any] = {
            "algorithm":    results[0].algorithm,
            "n_seeds":      len(results),
            "n_satellites": results[0].n_satellites,
            "battery":      self._battery_metrics(results),
            "mission":      self._mission_metrics(results),
            "safety":       self._safety_metrics(results),
            "learning":     self._learning_metrics(results),
            "meta":         self._meta_metrics(results),
        }
        return summary

    # ── Battery Metrics ───────────────────────────────────────────────────────

    def _battery_metrics(self, results: List[EpisodeResult]) -> Dict[str, float]:
        """
        Metrics:
          soc_mean_min     : mean of per-episode minimum SoC across satellites
          soc_std          : std of per-step mean SoC (stability measure)
          soh_final_mean   : mean final SoH (higher = less degradation)
          battery_lifetime : median episodes until SoH < threshold (extrapolated)
          dod_mean         : mean Depth of Discharge per cycle
        """
        soc_min_vals  = []
        soc_stds      = []
        soh_finals    = []
        dod_vals      = []

        for ep in results:
            if not ep.soc_trajectory:
                continue
            soc_arr = np.array(ep.soc_trajectory)    # (T, n_sat)
            soc_mean_t  = soc_arr.mean(axis=1)        # (T,) mean across sats
            soc_min_vals.append(float(soc_arr.min()))
            soc_stds.append(float(soc_mean_t.std()))

            if ep.soh_trajectory:
                soh_arr = np.array(ep.soh_trajectory)  # (T, n_sat)
                soh_finals.append(float(soh_arr[-1].mean()))
                # DoD = range of SoC over the episode for each sat
                dod = float((soc_arr.max(axis=0) - soc_arr.min(axis=0)).mean())
                dod_vals.append(dod)

        # Estimate battery lifetime in episodes:
        # If SoH degrades at rate Δ per episode → lifetime ≈ (1 - threshold) / Δ
        lifetime_est = np.nan
        if len(soh_finals) > 1:
            soh0 = 1.0
            soh_final_mean = float(np.mean(soh_finals))
            delta_per_ep   = max(soh0 - soh_final_mean, 1e-9)
            lifetime_est   = (soh0 - self.soh_eol_thr) / delta_per_ep

        return {
            "soc_mean_min":      _stat(soc_min_vals),
            "soc_stability_std": _stat(soc_stds),
            "soh_final":         _stat(soh_finals),
            "battery_lifetime_est_episodes": float(lifetime_est),
            "dod_mean":          _stat(dod_vals),
        }

    # ── Mission Metrics ───────────────────────────────────────────────────────

    def _mission_metrics(self, results: List[EpisodeResult]) -> Dict[str, float]:
        """
        Metrics:
          mission_success_rate   : tasks_completed / tasks_scheduled
          tasks_per_orbit        : tasks completed per orbit period
          coordination_efficiency: how close to optimal task allocation
          reward_mean            : mean total episode reward
          reward_std             : std of total episode reward
        """
        success_rates    = []
        tasks_per_orbits = []
        rewards          = [ep.total_reward for ep in results]

        for ep in results:
            if ep.tasks_scheduled > 0:
                rate = ep.tasks_completed / ep.tasks_scheduled
                success_rates.append(rate)
            n_orbits = max(ep.episode_length / self.steps_per_orbit, 1e-9)
            tasks_per_orbits.append(ep.tasks_completed / n_orbits)

        # coordination_efficiency: payload_ON fraction (action=0) weighted by SoC safety
        coord_eff_vals = []
        for ep in results:
            if not ep.action_trajectory:
                continue
            actions = np.array(ep.action_trajectory)  # (T, n_sat)
            payload_frac = float((actions == 0).mean())  # fraction of payload_ON steps
            coord_eff_vals.append(payload_frac)

        return {
            "mission_success_rate": _stat(success_rates),
            "tasks_per_orbit":      _stat(tasks_per_orbits),
            "coordination_eff":     _stat(coord_eff_vals),
            "reward_mean":          float(np.mean(rewards)),
            "reward_std":           float(np.std(rewards)),
            "reward_max":           float(np.max(rewards)),
        }

    # ── Safety Metrics ────────────────────────────────────────────────────────

    def _safety_metrics(self, results: List[EpisodeResult]) -> Dict[str, float]:
        """
        Metrics:
          thermal_violations_per_ep : mean thermal violation events per episode
          soc_critical_events_per_ep: mean SoC-critical events per episode
          safety_override_rate      : fraction of steps where safety gate fired
          recovery_success_rate     : fraction of critical events resolved (no episode termination)
        """
        thermal_counts   = [len(ep.thermal_viol_steps)    for ep in results]
        soc_crit_counts  = [len(ep.soc_critical_steps)    for ep in results]
        override_counts  = [len(ep.safety_override_steps) for ep in results]
        ep_lengths       = [max(ep.episode_length, 1)     for ep in results]

        override_rates   = [o / l for o, l in zip(override_counts, ep_lengths)]

        # Recovery success: episode didn't terminate early due to SoC=0
        expected_len = results[0].episode_length if results else 1000
        recovery_flags = [int(ep.episode_length >= 0.9 * expected_len) for ep in results]

        return {
            "thermal_violations_per_ep":  float(np.mean(thermal_counts)),
            "soc_critical_events_per_ep": float(np.mean(soc_crit_counts)),
            "safety_override_rate":       float(np.mean(override_rates)),
            "recovery_success_rate":      float(np.mean(recovery_flags)),
        }

    # ── Learning Metrics ──────────────────────────────────────────────────────

    def _learning_metrics(self, results: List[EpisodeResult]) -> Dict[str, float]:
        """
        Metrics derived from reward trajectory shapes (per-episode).
          reward_convergence_step: first episode where reward > 90% of peak
          reward_stability_last20: std of last 20% of per-step rewards
        """
        # Per-step reward stats across episodes
        all_step_rewards = []
        for ep in results:
            if ep.reward_trajectory:
                all_step_rewards.append(np.array(ep.reward_trajectory))

        if not all_step_rewards:
            return {"reward_step_mean": np.nan, "reward_step_std": np.nan,
                    "reward_stability_last20pct": np.nan}

        min_len = min(len(r) for r in all_step_rewards)
        reward_mat = np.stack([r[:min_len] for r in all_step_rewards], axis=0)  # (n_ep, T)

        mean_curve = reward_mat.mean(axis=0)   # (T,)
        tail_len   = max(int(0.2 * min_len), 1)
        tail_std   = float(mean_curve[-tail_len:].std())

        peak        = float(mean_curve.max())
        threshold   = 0.9 * peak
        conv_step   = int(np.argmax(mean_curve >= threshold)) if peak > 0 else -1

        return {
            "reward_step_mean":          float(mean_curve.mean()),
            "reward_step_std":           float(mean_curve.std()),
            "reward_stability_last20pct": tail_std,
            "convergence_step":          conv_step,
            "peak_step_reward":          peak,
        }

    # ── Meta Metrics ──────────────────────────────────────────────────────────

    def _meta_metrics(self, results: List[EpisodeResult]) -> Dict[str, Any]:
        return {
            "n_episodes":          len(results),
            "mean_episode_length": float(np.mean([ep.episode_length for ep in results])),
            "algorithm":           results[0].algorithm,
            "n_satellites":        results[0].n_satellites,
        }

    # ── Output Utilities ──────────────────────────────────────────────────────

    def print_table(self, summary: Dict[str, Any]) -> None:
        """Print a formatted metric table (suitable for copy-paste into papers)."""
        algo = summary.get("algorithm", "Unknown")
        n_s  = summary.get("n_satellites", "?")
        n_seeds = summary.get("n_seeds", "?")

        print(f"\n{'='*65}")
        print(f"  CASC-RL Evaluation Results: {algo} | {n_s} satellites | {n_seeds} seeds")
        print(f"{'='*65}")

        sections = {
            "BATTERY": summary.get("battery", {}),
            "MISSION": summary.get("mission", {}),
            "SAFETY":  summary.get("safety",  {}),
            "LEARNING": summary.get("learning", {}),
        }
        for section, metrics in sections.items():
            print(f"\n  [{section}]")
            for k, v in metrics.items():
                if isinstance(v, dict):
                    mean = v.get("mean", np.nan)
                    std  = v.get("std",  np.nan)
                    print(f"    {k:<40s}: {mean:.4f} ± {std:.4f}")
                elif isinstance(v, float):
                    print(f"    {k:<40s}: {v:.4f}")
                else:
                    print(f"    {k:<40s}: {v}")
        print(f"{'='*65}\n")

    def save_json(self, summary: Dict[str, Any], path: str) -> None:
        """Save metrics summary to JSON for downstream analysis."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(_make_serializable(summary), f, indent=2)
        print(f"Metrics saved to: {path}")

    def save_csv(self, summaries: List[Dict[str, Any]], path: str) -> None:
        """
        Save multiple algorithm summaries as a flat CSV table.
        Suitable for pasting directly into LaTeX or Excel.
        """
        rows = []
        for s in summaries:
            row = {"algorithm": s["algorithm"], "n_satellites": s["n_satellites"]}
            for section in ("battery", "mission", "safety", "learning"):
                for k, v in s.get(section, {}).items():
                    if isinstance(v, dict):
                        row[f"{section}.{k}.mean"] = v.get("mean", "")
                        row[f"{section}.{k}.std"]  = v.get("std",  "")
                    else:
                        row[f"{section}.{k}"] = v
            rows.append(row)

        if not rows:
            return
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Comparison table saved to: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Paper-Specific Metrics (individual functions for plotting)
# ─────────────────────────────────────────────────────────────────────────────

def compute_soc_trajectory_stats(
    results: List[EpisodeResult],
) -> Dict[str, np.ndarray]:
    """
    Compute mean ± std SoC trajectory across seeds and satellites.
    Returns arrays of shape (T,) usable directly for matplotlib fill_between plots.
    """
    trajs = []
    for ep in results:
        if ep.soc_trajectory:
            arr = np.array(ep.soc_trajectory)   # (T, n_sat)
            trajs.append(arr.mean(axis=1))       # (T,) mean SoC across sats

    if not trajs:
        return {}
    min_len = min(len(t) for t in trajs)
    mat     = np.stack([t[:min_len] for t in trajs], axis=0)  # (n_seeds, T)
    return {
        "mean": mat.mean(axis=0),
        "std":  mat.std(axis=0),
        "min":  mat.min(axis=0),
        "max":  mat.max(axis=0),
    }


def compute_soh_degradation_rate(results: List[EpisodeResult]) -> Dict[str, float]:
    """
    Compute mean SoH degradation rate (ΔSoH per 1000 steps).
    """
    rates = []
    for ep in results:
        if ep.soh_trajectory and ep.episode_length > 0:
            soh_arr = np.array(ep.soh_trajectory)    # (T, n_sat)
            delta   = float(soh_arr[0].mean() - soh_arr[-1].mean())
            rate    = delta / ep.episode_length * 1000  # per 1000 steps
            rates.append(rate)
    return _stat_dict(rates)


def compute_mission_success_per_orbit(results: List[EpisodeResult], steps_per_orbit: int = 552) -> Dict[str, float]:
    """Tasks completed per orbital pass (n_tasks_completed / n_orbits)."""
    vals = []
    for ep in results:
        n_orbits = max(ep.episode_length / steps_per_orbit, 1e-9)
        vals.append(ep.tasks_completed / n_orbits)
    return _stat_dict(vals)


def compute_thermal_violation_rate(results: List[EpisodeResult]) -> Dict[str, float]:
    """Thermal violations per 1000 simulation steps."""
    vals = []
    for ep in results:
        if ep.episode_length > 0:
            rate = len(ep.thermal_viol_steps) / ep.episode_length * 1000
            vals.append(rate)
    return _stat_dict(vals)


def compute_safety_override_rate(results: List[EpisodeResult]) -> Dict[str, float]:
    """Fraction of steps where safety gate overrode the policy action."""
    vals = []
    for ep in results:
        if ep.episode_length > 0:
            vals.append(len(ep.safety_override_steps) / ep.episode_length)
    return _stat_dict(vals)


def compute_reward_curve(
    results: List[EpisodeResult],
    smoothing: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Compute smoothed mean reward trajectory across seeds.
    Returns dict with 'mean', 'std', 'steps' arrays of shape (T,).
    """
    trajs = [ep.reward_trajectory for ep in results if ep.reward_trajectory]
    if not trajs:
        return {}
    min_len = min(len(t) for t in trajs)
    mat = np.stack([t[:min_len] for t in trajs], axis=0)  # (n_seeds, T)

    def _smooth(x: np.ndarray) -> np.ndarray:
        kernel = np.ones(smoothing) / smoothing
        return np.convolve(x, kernel, mode="valid")

    mat_smooth = np.stack([_smooth(mat[i]) for i in range(len(mat))])
    return {
        "mean":  mat_smooth.mean(axis=0),
        "std":   mat_smooth.std(axis=0),
        "steps": np.arange(len(mat_smooth[0])),
    }


def build_comparison_table(
    all_results: Dict[str, List[EpisodeResult]],
    n_satellites: int = 3,
) -> List[Dict[str, Any]]:
    """
    Build paper comparison table across algorithms.

    Args:
        all_results: {algorithm_name: [EpisodeResult, ...], ...}
        n_satellites: constellation size for labelling

    Returns:
        List of row dicts, one per algorithm — ready for MetricComputer.save_csv()
    """
    mc   = MetricComputer()
    rows = []
    for algo_name, results in all_results.items():
        # Tag each result with the algorithm name
        for r in results:
            r.algorithm = algo_name
        row = mc.compute_all(results)
        row["algorithm"]    = algo_name
        row["n_satellites"] = n_satellites
        rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Significance Testing
# ─────────────────────────────────────────────────────────────────────────────

def welch_t_test(
    a: List[float],
    b: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Welch's two-sided t-test for two independent samples.
    Used to confirm statistical significance of CASC-RL vs. baselines.

    Returns:
        {"t_stat", "p_value", "significant", "effect_size" (Cohen's d)}
    """
    from scipy import stats            # lazy import — optional dependency
    a_arr, b_arr = np.array(a), np.array(b)
    t, p     = stats.ttest_ind(a_arr, b_arr, equal_var=False)
    pooled_std = np.sqrt((a_arr.std()**2 + b_arr.std()**2) / 2)
    cohen_d  = (a_arr.mean() - b_arr.mean()) / (pooled_std + 1e-12)
    return {
        "t_stat":      float(t),
        "p_value":     float(p),
        "significant": bool(p < alpha),
        "effect_size_cohens_d": float(cohen_d),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stat(vals: List[float]) -> Dict[str, float]:
    """Return {mean, std, min, max} dict for a list of floats."""
    if not vals:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    a = np.array(vals, dtype=float)
    return {
        "mean": float(a.mean()),
        "std":  float(a.std()),
        "min":  float(a.min()),
        "max":  float(a.max()),
    }


def _stat_dict(vals: List[float]) -> Dict[str, float]:
    return _stat(vals)


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python scalars for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
