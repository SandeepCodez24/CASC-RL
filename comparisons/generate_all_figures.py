"""
generate_all_figures.py — One-Command Figure Generation.

Runs every comparison and summary plot script to produce all paper figures.
Saves PDF + PNG to figures/ directory.

Usage:
    python comparisons/generate_all_figures.py
    python comparisons/generate_all_figures.py --figures_dir my_figures --results_dir results
"""
from __future__ import annotations
import argparse, os, sys, time
from loguru import logger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate all CASC-RL paper figures")
    p.add_argument("--results_dir",  default="results")
    p.add_argument("--figures_dir",  default="figures")
    p.add_argument("--n_satellites", type=int, default=3)
    args = p.parse_args()

    kw = dict(results_dir=args.results_dir, figures_dir=args.figures_dir,
              n_satellites=args.n_satellites)

    os.makedirs(args.figures_dir, exist_ok=True)
    t0 = time.time()
    errors = []

    PLOTS = [
        # Learning
        ("comparisons.learning.reward_vs_episode_comparison",    "plot_reward_curves",          kw),
        ("comparisons.learning.sample_efficiency_comparison",    "plot_sample_efficiency",       kw),
        ("comparisons.learning.convergence_speed_comparison",    "plot_convergence_speed",       kw),
        # Battery
        ("comparisons.battery.soc_trajectory_comparison",        "plot_soc_trajectories",        kw),
        ("comparisons.battery.soh_degradation_comparison",       "plot_soh_degradation",         kw),
        ("comparisons.battery.soc_min_stress_comparison",        "plot_soc_min_stress",          kw),
        ("comparisons.battery.battery_lifetime_comparison",      "plot_battery_lifetime",        kw),
        # Mission
        ("comparisons.mission.mission_success_rate_comparison",  "plot_mission_success_rate",    kw),
        ("comparisons.mission.task_completion_comparison",       "plot_task_completion",         kw),
        ("comparisons.mission.coordination_efficiency_comparison","plot_coordination_efficiency", kw),
        # Safety
        ("comparisons.safety.thermal_violations_comparison",     "plot_thermal_violations",      kw),
        ("comparisons.safety.safety_events_comparison",          "plot_safety_events",           kw),
        ("comparisons.safety.fault_recovery_comparison",         "plot_fault_recovery",          kw),
        # World Model
        ("comparisons.world_model.prediction_accuracy_comparison","plot_prediction_accuracy",
            {"results_dir": args.results_dir, "figures_dir": args.figures_dir, 
             "wm_path": "checkpoints/world_model_best.pt"}),
        ("comparisons.world_model.horizon_error_comparison",     "plot_horizon_error",
            dict(results_dir=args.results_dir, figures_dir=args.figures_dir)),
        ("comparisons.world_model.eclipse_prediction_comparison","plot_eclipse_prediction",
            dict(results_dir=args.results_dir, figures_dir=args.figures_dir)),
        # Ablations
        ("comparisons.ablation.world_model_ablation",            "plot_world_model_ablation",
            dict(figures_dir=args.figures_dir)),
        ("comparisons.ablation.cooperative_layer_ablation",      "plot_cooperative_ablation",
            dict(figures_dir=args.figures_dir)),
        ("comparisons.ablation.hierarchical_layer_ablation",     "plot_hierarchical_ablation",
            dict(figures_dir=args.figures_dir)),
        ("comparisons.ablation.layer_contribution_ablation",     "plot_layer_contribution",
            dict(figures_dir=args.figures_dir)),
        # Scaling
        ("comparisons.scaling.constellation_size_scaling",       "plot_scaling",
            dict(results_dir=args.results_dir + "/scaling", figures_dir=args.figures_dir)),
        ("comparisons.scaling.compute_cost_scaling",             "plot_compute_scaling",
            dict(results_dir=args.results_dir, figures_dir=args.figures_dir)),
        # Summary
        ("comparisons.summary.radar_chart_all_methods",          "plot_radar",
            dict(figures_dir=args.figures_dir)),
        ("comparisons.summary.improvement_heatmap",              "plot_improvement_heatmap",
            dict(figures_dir=args.figures_dir)),
        ("comparisons.summary.overall_metrics_table",            "run",
            dict(results_dir=args.results_dir, figures_dir=args.figures_dir,
                 n_satellites=args.n_satellites)),
    ]

    logger.info(f"Generating {len(PLOTS)} figures → {args.figures_dir}/")
    for module_path, fn_name, kwargs in PLOTS:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            fn  = getattr(mod, fn_name)
            fn(**kwargs)
            logger.info(f"  ✓  {fn_name}")
        except Exception as e:
            logger.warning(f"  ✗  {fn_name}: {e}")
            errors.append((fn_name, str(e)))

    elapsed = time.time() - t0
    logger.success(
        f"\nFigure generation complete in {elapsed:.1f}s | "
        f"{len(PLOTS) - len(errors)}/{len(PLOTS)} succeeded | "
        f"Results: {args.figures_dir}/"
    )
    if errors:
        logger.warning("Failed plots:")
        for name, err in errors:
            logger.warning(f"  {name}: {err}")


if __name__ == "__main__":
    main()
