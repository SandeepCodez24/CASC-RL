"""
experiment_runner.py — Main Evaluation & Benchmarking Orchestrator.

Runs all baselines and CASC-RL across multiple seeds and constellation sizes.
Produces:
  1. JSON results files per algorithm / constellation size
  2. Paper comparison table (CSV) as in PROJECT_DOCUMENT §8
  3. Statistical significance report (Welch t-test, p-values)
  4. Per-metric matplotlib plots saved to figures/

Usage:
    python evaluation/experiment_runner.py
    python evaluation/experiment_runner.py --n_episodes 20 --n_satellites 3
    python evaluation/experiment_runner.py --quick  (3 episodes, for CI smoke test)
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import time
from typing import Dict, List, Any

# Project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from loguru import logger

from evaluation.metrics import (
    MetricComputer,
    EpisodeResult,
    build_comparison_table,
    welch_t_test,
    compute_soc_trajectory_stats,
    compute_reward_curve,
)
from evaluation.baseline_rule import run_rule_based_episodes
from evaluation.baseline_pid  import run_pid_episodes


# ─────────────────────────────────────────────────────────────────────────────
# CASC-RL Agent Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_casc_rl_episodes(
    n_episodes:      int,
    n_satellites:    int,
    episode_length:  int,
    enable_eclipse:  bool,
    seeds:           List[int],
    world_model_path: str,
    checkpoint_path:  str,
    use_mpc:         bool = False,
    verbose:         bool = False,
) -> List[EpisodeResult]:
    """
    Run CASC-RL (MAPPO + world model) evaluation episodes.

    Args:
        use_mpc: if True, use cognitive_decision() MPC loop instead of learned actor
    """
    import torch
    from environment.constellation_env import ConstellationEnv
    from world_model.world_model import WorldModel
    from agents.satellite_agent import SatelliteAgent
    from agents.policy_network import ActorNetwork
    from agents.critic_network import CriticNetwork
    from agents.action_selector import ActionSelector

    device = "cpu"

    wm = WorldModel(state_dim=8, action_dim=5, device=device)
    if os.path.exists(world_model_path):
        wm.load(world_model_path)
        logger.info(f"World model loaded from {world_model_path}")
    else:
        logger.warning("No world model checkpoint — using untrained model.")

    actors  = [ActorNetwork(state_dim=8, n_actions=5, predict_k=5) for _ in range(n_satellites)]
    critics = [CriticNetwork(state_dim=8) for _ in range(n_satellites)]
    selectors = [ActionSelector(agent_id=i) for i in range(n_satellites)]

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        # Handle both list-of-actors and centralized-critic schemas
        if "actors" in ckpt:
            for i, sd in enumerate(ckpt["actors"]):
                if i < len(actors): actors[i].load_state_dict(sd)
        
        # If the checkpoint is from the curriculum trainer (list mapping)
        for i, actor in enumerate(actors):
            key = f"actor_{i}"
            if key in ckpt:
                actor.load_state_dict(ckpt[key])
        logger.info(f"Agents loaded from {checkpoint_path}")
    else:
        logger.warning("No agent checkpoint — using random policy.")

    agents  = [
        SatelliteAgent(
            agent_id=i, 
            actor=actors[i], 
            critic=critics[i], 
            world_model=wm, 
            action_selector=selectors[i]
        ) 
        for i in range(n_satellites)
    ]
    algo    = "CASC-RL-MPC" if use_mpc else "CASC-RL"
    results = []

    IDX_SOC, IDX_SOH, IDX_TEMP = 0, 1, 2
    SOC_CRIT_THR, THERMAL_THR  = 0.15, 0.60

    for ep_idx, seed in enumerate(seeds[:n_episodes]):
        env = ConstellationEnv(n_satellites=n_satellites, enable_eclipse=enable_eclipse)
        obs, _ = env.reset(seed=seed)

        result = EpisodeResult(
            episode_id=ep_idx, n_satellites=n_satellites,
            seed=seed, total_reward=0.0, algorithm=algo,
        )

        for step in range(episode_length):
            actions = []
            for i, agent in enumerate(agents):
                if use_mpc:
                    action, _ = agent.cognitive_decision(obs[i], k=5)
                else:
                    action, _, _, info = agent.act(obs[i], deterministic=True)
                    if info.get("was_overridden"):
                        result.safety_override_steps.append(step)
                actions.append(action)

            next_obs, rewards, terminated, truncated, info = env.step(actions)
            step_reward = float(rewards)

            result.total_reward      += step_reward
            result.reward_trajectory.append(step_reward)
            result.soc_trajectory.append(next_obs[:, IDX_SOC].copy())
            result.soh_trajectory.append(next_obs[:, IDX_SOH].copy())
            result.temp_trajectory.append(next_obs[:, IDX_TEMP].copy())
            result.action_trajectory.append(np.array(actions))

            if (next_obs[:, IDX_SOC] < SOC_CRIT_THR).any():
                result.soc_critical_steps.append(step)
            if (next_obs[:, IDX_TEMP] > THERMAL_THR).any():
                result.thermal_viol_steps.append(step)

            result.episode_length = step + 1
            obs = next_obs
            if terminated or truncated:
                break

        if verbose:
            logger.info(f"[{algo}] Ep {ep_idx+1}/{n_episodes}: reward={result.total_reward:.3f} | "
                        f"SoC_min={min(s.min() for s in result.soc_trajectory):.3f} | "
                        f"overrides={len(result.safety_override_steps)}")
        results.append(result)
        env.close()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main Experiment Runner
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentRunner:
    """
    Orchestrates the full benchmark suite for one constellation size.

    Runs all 4 algorithms:
      1. PID Controller (Baseline A)
      2. Rule-Based Scheduler (Baseline B)
      3. CASC-RL with learned actor (MAPPO)
      4. CASC-RL with MPC loop (Algorithm 4)

    Args:
        n_episodes:        number of evaluation seeds (default: 20)
        n_satellites:      constellation size (default: 3)
        episode_length:    steps per episode (default: 1000)
        results_dir:       directory to save JSON/CSV outputs
        figures_dir:       directory to save matplotlib figures
        world_model_path:  path to pre-trained world model
        checkpoint_path:   path to trained MAPPO checkpoint
        enable_eclipse:    toggle eclipse in evaluation
    """

    def __init__(
        self,
        n_episodes:       int  = 20,
        n_satellites:     int  = 3,
        episode_length:   int  = 1000,
        results_dir:      str  = "results",
        figures_dir:      str  = "figures",
        world_model_path: str  = "checkpoints/world_model_best.pt",
        checkpoint_path:  str  = "checkpoints/mappo_best.pt",
        enable_eclipse:   bool = True,
        verbose:          bool = True,
    ):
        self.n_episodes       = n_episodes
        self.n_satellites     = n_satellites
        self.episode_length   = episode_length
        self.results_dir      = results_dir
        self.figures_dir      = figures_dir
        self.world_model_path = world_model_path
        self.checkpoint_path  = checkpoint_path
        self.enable_eclipse   = enable_eclipse
        self.verbose          = verbose

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        self.seeds = list(range(n_episodes))

    # ── Public API ────────────────────────────────────────────────────────────

    def run_all(self) -> Dict[str, List[EpisodeResult]]:
        """
        Run all algorithms and return {algo_name: [EpisodeResult, ...]} dict.
        """
        logger.info(
            f"Starting benchmark: {self.n_episodes} episodes × {self.n_satellites} satellites"
        )
        all_results: Dict[str, List[EpisodeResult]] = {}

        # Baseline A: PID Controller
        logger.info("Running Baseline A: PID Controller ...")
        t0 = time.time()
        all_results["PID_Controller"] = run_pid_episodes(
            n_episodes=self.n_episodes, n_satellites=self.n_satellites,
            episode_length=self.episode_length, enable_eclipse=self.enable_eclipse,
            seeds=self.seeds, verbose=self.verbose,
        )
        logger.info(f"PID done in {time.time()-t0:.1f}s")

        # Baseline B: Rule-Based
        logger.info("Running Baseline B: Rule-Based Scheduler ...")
        t0 = time.time()
        all_results["RuleBasedScheduler"] = run_rule_based_episodes(
            n_episodes=self.n_episodes, n_satellites=self.n_satellites,
            episode_length=self.episode_length, enable_eclipse=self.enable_eclipse,
            seeds=self.seeds, verbose=self.verbose,
        )
        logger.info(f"Rule-based done in {time.time()-t0:.1f}s")

        # CASC-RL (MAPPO learned actor)
        logger.info("Running CASC-RL (MAPPO) ...")
        t0 = time.time()
        all_results["CASC-RL"] = run_casc_rl_episodes(
            n_episodes=self.n_episodes, n_satellites=self.n_satellites,
            episode_length=self.episode_length, enable_eclipse=self.enable_eclipse,
            seeds=self.seeds, world_model_path=self.world_model_path,
            checkpoint_path=self.checkpoint_path, use_mpc=False, verbose=self.verbose,
        )
        logger.info(f"CASC-RL done in {time.time()-t0:.1f}s")

        # CASC-RL MPC (Algorithm 4 explicit MPC)
        logger.info("Running CASC-RL-MPC (Algorithm 4) ...")
        t0 = time.time()
        all_results["CASC-RL-MPC"] = run_casc_rl_episodes(
            n_episodes=self.n_episodes, n_satellites=self.n_satellites,
            episode_length=self.episode_length, enable_eclipse=self.enable_eclipse,
            seeds=self.seeds, world_model_path=self.world_model_path,
            checkpoint_path=self.checkpoint_path, use_mpc=True, verbose=self.verbose,
        )
        logger.info(f"CASC-RL-MPC done in {time.time()-t0:.1f}s")

        return all_results

    def compute_and_save_metrics(
        self, all_results: Dict[str, List[EpisodeResult]]
    ) -> List[Dict[str, Any]]:
        """Compute metrics for all algorithms and save results."""
        mc   = MetricComputer()
        rows = []
        for algo, results in all_results.items():
            summary = mc.compute_all(results)
            summary["algorithm"]    = algo
            summary["n_satellites"] = self.n_satellites
            rows.append(summary)

            # Individual JSON per algorithm
            json_path = os.path.join(self.results_dir, f"{algo}_n{self.n_satellites}.json")
            mc.save_json(summary, json_path)
            if self.verbose:
                mc.print_table(summary)

        # Combined CSV table (paper-ready)
        csv_path = os.path.join(self.results_dir, f"comparison_n{self.n_satellites}.csv")
        mc.save_csv(rows, csv_path)
        logger.info(f"Comparison table saved: {csv_path}")
        return rows

    def run_significance_tests(
        self, all_results: Dict[str, List[EpisodeResult]]
    ) -> Dict[str, Any]:
        """
        Run Welch t-tests: CASC-RL vs. each baseline, on total_reward.
        """
        casc_rewards = [ep.total_reward for ep in all_results.get("CASC-RL", [])]
        report       = {}

        for algo, results in all_results.items():
            if algo == "CASC-RL":
                continue
            other_rewards = [ep.total_reward for ep in results]
            if len(other_rewards) < 2 or len(casc_rewards) < 2:
                continue
            test = welch_t_test(casc_rewards, other_rewards)
            report[f"CASC-RL_vs_{algo}"] = test
            sig = "✓ significant" if test["significant"] else "✗ not significant"
            logger.info(
                f"CASC-RL vs {algo}: p={test['p_value']:.4f} | "
                f"d={test['effect_size_cohens_d']:.3f} | {sig}"
            )

        sig_path = os.path.join(self.results_dir, f"significance_n{self.n_satellites}.json")
        os.makedirs(self.results_dir, exist_ok=True)
        with open(sig_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Significance tests saved: {sig_path}")
        return report

    def save_trajectory_data(self, all_results: Dict[str, List[EpisodeResult]]) -> None:
        """Save SoC and reward trajectory arrays for plotting scripts."""
        for algo, results in all_results.items():
            soc_stats = compute_soc_trajectory_stats(results)
            rew_stats = compute_reward_curve(results, smoothing=20)

            out = {
                "soc": {k: v.tolist() for k, v in soc_stats.items() if isinstance(v, np.ndarray)},
                "reward": {k: v.tolist() for k, v in rew_stats.items() if isinstance(v, np.ndarray)},
            }
            path = os.path.join(self.results_dir, f"trajectories_{algo}_n{self.n_satellites}.json")
            with open(path, "w") as f:
                json.dump(out, f)
            logger.info(f"Trajectory data saved: {path}")

    def run_full_benchmark(self) -> None:
        """Entry point: run all, compute metrics, significance tests, save data."""
        t_total = time.time()

        all_results = self.run_all()
        self.compute_and_save_metrics(all_results)
        self.run_significance_tests(all_results)
        self.save_trajectory_data(all_results)

        elapsed = (time.time() - t_total) / 60
        logger.success(
            f"Full benchmark complete in {elapsed:.1f} min. "
            f"Results: {self.results_dir}/"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Scaling Experiment (3 → 6 → 12 satellites)
# ─────────────────────────────────────────────────────────────────────────────

def run_scaling_experiment(
    sizes:            List[int] = [3, 6, 12],
    n_episodes:       int       = 10,
    episode_length:   int       = 1000,
    results_dir:      str       = "results/scaling",
    figures_dir:      str       = "figures",
    world_model_path: str       = "checkpoints/world_model_best.pt",
) -> None:
    """
    Run evaluation at multiple constellation sizes and save scaling results.
    Used to generate the scaling plot (Figure 6 in the paper).
    """
    os.makedirs(results_dir, exist_ok=True)
    scaling_data = {}

    for n_sat in sizes:
        logger.info(f"\n{'='*50}\nScaling: {n_sat} satellites\n{'='*50}")

        ckpt_path = f"checkpoints/curriculum/stage_best_{n_sat}sat.pt"
        if not os.path.exists(ckpt_path):
            ckpt_path = "checkpoints/mappo_best.pt"

        runner = ExperimentRunner(
            n_episodes=n_episodes, n_satellites=n_sat,
            episode_length=episode_length, results_dir=results_dir,
            world_model_path=world_model_path, checkpoint_path=ckpt_path, verbose=False,
        )
        all_results = runner.run_all()
        rows        = runner.compute_and_save_metrics(all_results)
        scaling_data[str(n_sat)] = rows

    # Save scaling summary
    path = os.path.join(results_dir, "scaling_summary.json")
    import json
    with open(path, "w") as f:
        def _ser(obj):
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _ser(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_ser(v) for v in obj]
            return obj
        json.dump(_ser(scaling_data), f, indent=2)
    logger.success(f"Scaling experiment complete. Results: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CASC-RL Experiment Runner")
    p.add_argument("--n_episodes",       type=int,  default=20)
    p.add_argument("--n_satellites",     type=int,  default=3)
    p.add_argument("--episode_length",   type=int,  default=1000)
    p.add_argument("--results_dir",      type=str,  default="results")
    p.add_argument("--figures_dir",      type=str,  default="figures")
    p.add_argument("--world_model_path", type=str,  default="checkpoints/world_model_best.pt")
    p.add_argument("--checkpoint_path",  type=str,  default="checkpoints/mappo_best.pt")
    p.add_argument("--scaling",          action="store_true",
                   help="Run scaling experiment across 3/6/12 satellite configs")
    p.add_argument("--quick",            action="store_true",
                   help="Quick smoke test: 3 episodes, 100 steps")
    p.add_argument("--no_eclipse",       action="store_true")
    p.add_argument("--verbose",          action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        args.n_episodes     = 3
        args.episode_length = 100
        logger.info("Quick mode: 3 episodes, 100 steps each")

    if args.scaling:
        run_scaling_experiment(
            sizes=[3, 6, 12],
            n_episodes=args.n_episodes,
            episode_length=args.episode_length,
            results_dir=os.path.join(args.results_dir, "scaling"),
            world_model_path=args.world_model_path,
        )
    else:
        runner = ExperimentRunner(
            n_episodes=args.n_episodes,
            n_satellites=args.n_satellites,
            episode_length=args.episode_length,
            results_dir=args.results_dir,
            figures_dir=args.figures_dir,
            world_model_path=args.world_model_path,
            checkpoint_path=args.checkpoint_path,
            enable_eclipse=not args.no_eclipse,
            verbose=args.verbose,
        )
        runner.run_full_benchmark()


if __name__ == "__main__":
    main()
