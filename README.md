# CASC-RL: Cognitive Autonomous Satellite Constellation with Reinforcement Learning

> **Research-Grade AI/ML System** | Aerospace · Multi-Agent RL · Digital Twin

## Overview

CASC-RL is a research framework combining physics-accurate Digital Twin simulation, Predictive World Models, and Cooperative MARL (MAPPO) for autonomous satellite constellation management.

## Quickstart

```bash
# 1. Clone repository
git clone https://github.com/your-org/casc-rl.git
cd casc-rl

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify environment
python -c "from environment.constellation_env import ConstellationEnv; print('OK')"

# 4. Train world model (Phase 2)
python training/train_world_model.py

# 5. Train MARL (Phase 4)
python training/train_marl.py

# 6. Run evaluation
python evaluation/experiment_runner.py

# 7. Launch dashboard
python visualization/dashboard.py
```

## Project Structure

```
casc-rl/
├── config/          # YAML configuration files
├── environment/     # Layer 1: Physics/Digital Twin simulation
├── world_model/     # Layer 2a: Predictive Neural World Model
├── agents/          # Layer 2b: Per-satellite RL agents
├── marl/            # Layer 3: Cooperative MAPPO training
├── coordination/    # Layer 4: Hierarchical Coordination
├── safety/          # Layer 5: Anomaly detection & recovery
├── training/        # Training pipeline scripts
├── evaluation/      # Benchmarking & metrics
├── visualization/   # Plots & interactive dashboard
├── experiments/     # Experiment scenarios
├── tests/           # Unit & integration tests
├── docs/            # Documentation
└── api/             # Ground station REST API
```

## Architecture

| Layer | Component | Algorithm |
|---|---|---|
| L1: Physical | Digital Twin | Orbital, Power, Thermal Models |
| L2: Cognitive | Per-Satellite Agent | World Model + RL Policy |
| L3: Cooperative | Multi-Agent Learning | MAPPO |
| L4: Coordination | Task Allocation | Cluster Coordinator |
| L5: Recovery | Safety & Fault Handling | Anomaly Detector |

## Makefile Commands

```bash
make train-all    # Full training pipeline
make evaluate     # Run all evaluation scenarios
make dashboard    # Launch visualization dashboard
make test         # Run test suite
```

## Citation

```bibtex
@misc{casc-rl-2026,
  title  = {CASC-RL: Cognitive Autonomous Satellite Constellation with Reinforcement Learning},
  author = {[Author Names]},
  year   = {2026},
  note   = {GitHub repository: https://github.com/[your-repo]/casc-rl}
}
```

---
*See [PROJECT_DOCUMENT.md](PROJECT_DOCUMENT.md) for the full technical specification.*
