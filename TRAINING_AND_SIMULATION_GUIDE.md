# CASC-RL — Complete Local Training & Simulation Guide
### Step-by-Step: Setup → Train → Simulate → Evaluate

> All commands are written for **Windows PowerShell** and should be run from
> `d:\AIandDS_HUB\Projects\CASC-RL` (your project root) unless stated otherwise.
> Every section builds on the previous one — do not skip steps.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Environment Setup (One-Time)](#2-environment-setup-one-time)
3. [Project Structure Overview](#3-project-structure-overview)
4. [Phase 1 — Collect Data and Train the World Model](#4-phase-1--collect-data-and-train-the-world-model)
5. [Phase 2 — Train MARL Agents (MAPPO)](#5-phase-2--train-marl-agents-mappo)
6. [Phase 3 — Curriculum Training (Full Pipeline)](#6-phase-3--curriculum-training-full-pipeline)
7. [Running Simulations](#7-running-simulations)
8. [Evaluating Trained Agents](#8-evaluating-trained-agents)
9. [Running Tests](#9-running-tests)
10. [Monitoring Training](#10-monitoring-training)
11. [Configuration Reference](#11-configuration-reference)
12. [Troubleshooting](#12-troubleshooting)
13. [Quick Reference Commands](#13-quick-reference-commands)

---

## 1. System Requirements

| Item | Minimum | Recommended |
|---|---|---|
| **OS** | Windows 10 64-bit | Windows 11 |
| **Python** | 3.10 | 3.11 |
| **RAM** | 8 GB | 16+ GB |
| **GPU** | None (CPU-only) | NVIDIA RTX 3060+ with CUDA 11.8+ |
| **Disk** | 5 GB free | 20 GB (for large datasets) |

> CPU-only training works for 1–3 satellites. For 6–12 satellites or the full curriculum,
> a GPU is strongly recommended for 5–10x speedup.

---

## 2. Environment Setup (One-Time)

### Step 2.1 — Verify Python Version

```powershell
python --version
```

**Why:** The project uses Python 3.10+ type-hint syntax. Earlier versions
will throw `SyntaxError` on list[int] or dict[str, Any] type annotations.

Expected output: `Python 3.10.x` or `Python 3.11.x`

---

### Step 2.2 — Create a Virtual Environment

```powershell
python -m venv .venv
```

**Why:** Isolates CASC-RL's dependencies (PyTorch, Gymnasium, etc.) from
your system Python to avoid version conflicts.

---

### Step 2.3 — Activate the Virtual Environment

```powershell
.venv\Scripts\Activate.ps1
```

**Why:** All subsequent `pip install` and `python` commands now run inside
`.venv`. You should see `(.venv)` appear in your prompt.

> If PowerShell blocks script execution, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

### Step 2.4 — Upgrade pip

```powershell
python -m pip install --upgrade pip
```

---

### Step 2.5 — Install All Dependencies

```powershell
pip install -r requirements.txt
```

**What gets installed and why:**

| Package | Purpose | Used In |
|---|---|---|
| `torch>=2.0` | Neural network training and GPU compute | All layers |
| `gymnasium>=0.29` | RL environment API (reset/step/render) | Layer 1 (`ConstellationEnv`) |
| `numpy>=1.24` | Array math, orbital calculations | All physics models |
| `scipy>=1.11` | MILP solver for constrained task allocation | Layer 4 (`TaskAllocator`) |
| `loguru>=0.7` | Structured, levelled logging | All modules |
| `tqdm>=4.65` | Progress bars during data collection | `DatasetBuilder` |
| `tensorboard>=2.14` | Training curve visualization | MAPPO trainer |
| `pyyaml>=6.0` | YAML config file loading | `config/` directory |
| `h5py>=3.9` | Binary dataset storage (.npz/.hdf5) | `DatasetBuilder` |
| `pytest>=7.4` | Unit tests | `tests/` directory |

> **GPU users (NVIDIA):** Install the CUDA-enabled PyTorch build instead:
> ```powershell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```
> Replace `cu118` with your CUDA version (`cu118` for CUDA 11.8, `cu121` for CUDA 12.1).

---

### Step 2.6 — Install the Package in Editable Mode

```powershell
pip install -e .
```

**Why:** Registers the `casc_rl` package so all imports like
`from environment.constellation_env import ConstellationEnv` resolve correctly
from any working directory.

---

### Step 2.7 — Verify Installation

```powershell
python -c "import torch; import gymnasium; print('OK — torch', torch.__version__)"
```

Expected: `OK — torch 2.x.x`

```powershell
python -c "from environment.constellation_env import ConstellationEnv; env = ConstellationEnv(3); print('Env OK:', env.observation_space)"
```

Expected: `Env OK: MultiDiscrete([5 5 5])` — confirms all Layer 1 modules load.

---

## 3. Project Structure Overview

```
CASC-RL/
|
+-- config/                      <- YAML configuration files
|   +-- environment.yaml         <- Satellite physics parameters
|   +-- constellation.yaml       <- Constellation topology (orbits, ISL)
|   +-- training.yaml            <- MAPPO hyperparameters, curriculum stages
|
+-- environment/                 <- Layer 1: Physical Simulation (Digital Twin)
|   +-- constellation_env.py     <- Main Gymnasium environment (entry point)
|   +-- orbital_dynamics.py      <- Keplerian orbit propagator + J2 perturbation
|   +-- eclipse_model.py         <- Eclipse shadow geometry (umbra/penumbra)
|   +-- solar_model.py           <- Solar power with cos(theta) sun tracking
|   +-- battery_model.py         <- SoC Coulomb counting + Peukert's law
|   +-- degradation_model.py     <- SoH decay: Wohler + Arrhenius temperature
|   +-- thermal_model.py         <- Nodal thermal model (heat balance equation)
|
+-- world_model/                 <- Layer 2a: Predictive World Model
|   +-- dynamics_network.py      <- EnsembleDynamicsNetwork (5 MLPs, residual)
|   +-- world_model.py           <- WorldModel interface + z-score normalizer
|   +-- dataset_builder.py       <- (s,a,s') transition data collector
|   +-- training.py              <- WorldModelTrainer (MSE loss, per-member Adam)
|
+-- agents/                      <- Layer 2b: Satellite Cognitive Layer
|   +-- satellite_agent.py       <- SatelliteAgent: act() + cognitive_decision() MPC
|   +-- policy_network.py        <- ActorNetwork (LayerNorm MLP + Categorical output)
|   +-- critic_network.py        <- Local + Centralized CriticNetwork
|   +-- action_selector.py       <- Safety gate (hard constraint override)
|
+-- marl/                        <- Layer 3: Cooperative Intelligence (MAPPO)
|   +-- mappo_trainer.py         <- MAPPO: centralized train, decentralized exec
|   +-- cooperative_rewards.py   <- Reward shaping: local + global blend
|   +-- advantage_estimator.py   <- GAE (gamma=0.99, lambda=0.95)
|   +-- buffer.py                <- Rollout buffer for PPO updates
|   +-- communication_protocol.py<- Global state construction for centralized critic
|
+-- coordination/                <- Layer 4: Hierarchical Coordination
|   +-- cluster_coordinator.py   <- ClusterCoordinator (Algorithm 5)
|   +-- task_allocator.py        <- Greedy + scipy MILP task assignment
|   +-- scheduling.py            <- PayloadScheduler (orbital period time slots)
|   +-- communication_protocol.py<- Layer 4 command broadcasting
|
+-- safety/                      <- Layer 5: Safety and Fault Recovery
|   +-- anomaly_detector.py      <- Z-score + world-model residual detection
|   +-- safety_monitor.py        <- FSM: NOMINAL->WARNING->CRITICAL->RECOVERY->DEGRADED
|   +-- recovery_policy.py       <- BatteryRecovery, ThermalRecovery, GeneralRecovery
|
+-- training/                    <- Training entry-point scripts
|   +-- train_world_model.py     <- Phase 1: Pre-train world model
|   +-- train_marl.py            <- Phase 2: MAPPO multi-agent training
|   +-- curriculum_training.py   <- Phase 3: Full 6-stage curriculum
|
+-- evaluation/                  <- Evaluation and benchmarking
|   +-- metrics.py               <- Performance metrics (SoC, mission success rate)
|   +-- experiment_runner.py     <- Evaluation harness
|
+-- tests/                       <- Unit tests for all modules
+-- checkpoints/                 <- Saved model weights (auto-created on first run)
+-- data/                        <- Transition datasets (auto-created)
+-- runs/                        <- TensorBoard training logs (auto-created)
+-- requirements.txt             <- All Python dependencies
+-- config/training.yaml         <- All hyperparameters
```

---

## 4. Phase 1 — Collect Data and Train the World Model

The World Model (`f_psi`) must be trained BEFORE the MARL policy. It learns to
predict `s_{t+1}` given `(s_t, a_t)` so that agents can perform k-step lookahead
without running the expensive physics simulation at inference time.

**File:** `training/train_world_model.py`
**Saves to:** `checkpoints/world_model_best.pt` and `checkpoints/world_model_final.pt`

---

### Step 4.1 — Run Default World Model Training

```powershell
python training\train_world_model.py
```

**What this script does internally:**

1. Creates `ConstellationEnv` with 3 satellites and eclipse enabled
2. Runs a random policy for 100,000 steps, saving `(s_t, a_t, s_{t+1})` tuples
   to `data/transitions.npz`
3. Builds `WorldModel` with 5 ensemble members (256-unit hidden layers, 3 layers each)
4. Fits a Z-score normalizer on collected states (Welford online algorithm)
5. Trains each ensemble member independently with Adam (lr=1e-3) for 100 epochs
6. Evaluates on 15% held-out validation set per epoch, logging per-dimension RMSE
7. Saves best checkpoint whenever validation loss improves

**Expected terminal output:**
```
Using device: cpu
Creating ConstellationEnv with 3 satellites (eclipse=ON) ...
Collecting 100,000 transitions (33,334 steps, 3 satellites)...
100%|####################| 33334/33334 [01:23<00:00, 398.3step/s]
Saved 100000 transitions to data/transitions.npz
Normalizer fitted on training data.
Epoch   10/100 | train=0.00842 | val=0.00891 | [SoC=0.0041 | SoH=0.0002 | temperature=0.0031 ...]
Epoch   20/100 | train=0.00631 | val=0.00712 | ...
...
Training complete. Best val loss: 0.00423
Final world model saved to: checkpoints/world_model_final.pt
World model training complete! Best val loss: 0.00423
```

**Time estimate:**
- CPU: 5–10 minutes
- GPU: 1–2 minutes

---

### Step 4.2 — Advanced Options for World Model Training

**Collect more data for better accuracy (recommended for 6+ satellites):**
```powershell
python training\train_world_model.py --n_transitions 200000 --n_epochs 150
```

**Use GPU:**
```powershell
python training\train_world_model.py --device cuda
```

**Train for a 6-satellite constellation:**
```powershell
python training\train_world_model.py --n_satellites 6 --n_transitions 300000
```

**Only train (reuse existing dataset, skip data collection):**
```powershell
python training\train_world_model.py --load_data --data_path data/transitions.npz
```

**All available flags:**
```
--n_satellites   INT    Number of satellites (default: 3)
--n_transitions  INT    Transitions to collect (default: 100000)
--n_epochs       INT    Training epochs (default: 100)
--batch_size     INT    Mini-batch size (default: 512)
--lr             FLOAT  Adam learning rate (default: 1e-3)
--n_ensemble     INT    Ensemble members (default: 5)
--hidden_dim     INT    Hidden layer width (default: 256)
--n_layers       INT    Hidden layers per member (default: 3)
--device         STR    cpu | cuda | auto (default: auto)
--data_path      STR    Dataset save/load path (default: data/transitions.npz)
--checkpoint_dir STR    Checkpoint directory (default: checkpoints/)
--load_data            Load existing dataset instead of collecting
--seed           INT    Random seed (default: 42)
```

---

### Step 4.3 — Verify World Model Quality

```powershell
python -c "
from world_model.world_model import WorldModel
import numpy as np

wm = WorldModel(state_dim=8, action_dim=5)
wm.load('checkpoints/world_model_best.pt')
print('World model loaded. Ensemble members:', wm.network.n_ensemble)

# Test prediction: relay_mode (action=3) for 5 steps
s_t  = np.array([0.8, 1.0, 0.2, 0.5, 0.3, 0.0, 0.25, 0.05], dtype='float32')
preds = wm.predict_k_steps(s_t, actions=[3]*5, k=5)
print('5-step SoC trajectory (relay mode drains battery):')
for i, s in enumerate(preds):
    print(f'  step {i+1}: SoC={s[0]:.4f}  Temp={s[2]:.4f}  SoH={s[1]:.4f}')
"
```

**Expected:** SoC values gradually decrease (relay mode uses 20W), temperature
is stable, SoH is near 1.0 (minimal degradation over 5 short steps).

---

## 5. Phase 2 — Train MARL Agents (MAPPO)

With the world model trained, train the multi-agent RL policy using MAPPO
(Multi-Agent Proximal Policy Optimization). Each satellite has its own actor.
A single centralized critic uses the global state (all satellites combined).

**File:** `training/train_marl.py`
**Requires:** `checkpoints/world_model_best.pt` from Phase 1
**Saves to:** `checkpoints/mappo_best.pt`, `checkpoints/mappo_final.pt`

---

### Step 5.1 — Run MAPPO Training (3 Satellites, Default Settings)

```powershell
python training\train_marl.py
```

**What this script does internally:**

1. Loads pre-trained world model from `checkpoints/world_model_best.pt`
2. Creates 3 `ActorNetwork` instances — one per satellite (LayerNorm MLP, 256 units)
3. Creates 1 `CentralizedCriticNetwork` — takes concatenated global state (3 x 8 = 24 dims)
4. Wraps each actor in a `SatelliteAgent` (integrates world model + actor + safety gate)
5. Runs 5,000 episodes, each 1,000 steps (= ~2.8 full orbits at dt=10s)
6. Every 200 steps: computes GAE advantages, runs 10 PPO update epochs
7. Saves best-episode checkpoint continuously

**The training algorithm per rollout (every 200 steps):**
```
Step 1: Collect 200 steps per episode for all agents
        Store (s_local, s_global, action, reward, done) in RolloutBuffer

Step 2: Compute shaped cooperative rewards via CooperativeRewardShaper
        shaped_r = 0.5 x mean(local_reward) + 0.5 x global_reward - 0.2 x conflict_penalty

Step 3: Compute GAE advantages using centralized critic
        delta_t = r_t + gamma * V(s_{t+1}) * (1-done) - V(s_t)
        A_t     = sum_l (gamma*lambda)^l * delta_{t+l}        [gamma=0.99, lambda=0.95]

Step 4: PPO clip update for each actor (10 epochs, 4 minibatches):
        ratio = exp(log_prob_new - log_prob_old)
        L_PPO = min(ratio * A, clip(ratio, 0.8, 1.2) * A) - 0.01 * entropy

Step 5: Update centralized critic with MSE on computed returns
```

**Expected terminal output:**
```
Device: cpu | Satellites: 3 | Episodes: 5000
Loaded world model from checkpoints/world_model_best.pt
Episode    50/5000 | avg_reward=0.412 | elapsed=2.3min
Episode   100/5000 | avg_reward=0.587 | elapsed=4.7min
Episode   500/5000 | avg_reward=0.891 | elapsed=23.1min
...
Episode  5000/5000 | avg_reward=1.243 | elapsed=48.2min
MARL training complete! Best episode reward: 1.389 | Total time: 48.2 min
```

**Time estimate:**
- CPU (3 satellites): 1–2 hours
- GPU (3 satellites): 15–30 minutes

---

### Step 5.2 — MAPPO Training Options

**GPU training:**
```powershell
python training\train_marl.py --device cuda
```

**More episodes with frequent logging:**
```powershell
python training\train_marl.py --n_episodes 10000 --log_every 100
```

**Train with 6 satellites:**
```powershell
python training\train_marl.py --n_satellites 6 --n_episodes 8000 --device cuda
```

**Custom hyperparameters (if default performance is poor):**
```powershell
python training\train_marl.py --lr_actor 1e-4 --entropy_coef 0.02 --clip_epsilon 0.15
```

**Save checkpoints more frequently:**
```powershell
python training\train_marl.py --save_every 200
```

**All available flags:**
```
--n_satellites       INT    Number of satellites (default: 3)
--n_episodes         INT    Training episodes (default: 5000)
--episode_length     INT    Steps per episode (default: 1000)
--rollout_length     INT    Steps before each PPO update (default: 200)
--n_epochs           INT    PPO update epochs per rollout (default: 10)
--lr_actor           FLOAT  Actor learning rate (default: 3e-4)
--lr_critic          FLOAT  Critic learning rate (default: 1e-3)
--gamma              FLOAT  Discount factor (default: 0.99)
--lam                FLOAT  GAE lambda (default: 0.95)
--clip_epsilon       FLOAT  PPO clip parameter (default: 0.2)
--entropy_coef       FLOAT  Entropy bonus weight (default: 0.01)
--batch_size         INT    Minibatch size (default: 512)
--device             STR    cpu | cuda | auto
--world_model_path   STR    World model checkpoint path
--checkpoint_dir     STR    Output checkpoint directory
--save_every         INT    Episodes between checkpoint saves (default: 500)
--log_every          INT    Episodes between log lines (default: 50)
```

---

## 6. Phase 3 — Curriculum Training (Full Pipeline)

Curriculum training progressively increases difficulty across 6 stages.
Each stage loads the checkpoint from the previous stage, transferring learned skills.

```
Stage 1  ->  Stage 2  ->  Stage 3  ->  Stage 4  ->  Stage 5  ->  Stage 6
1 sat        1 sat        3 sats       3 sats        6 sats       12 sats
no eclipse   eclipse      eclipse      + degrad.     stress       adversarial
1,000 ep.    2,000 ep.    2,000 ep.    2,000 ep.     3,000 ep.    5,000 ep.
```

**File:** `training/curriculum_training.py`
**Requires:** `checkpoints/world_model_best.pt` from Phase 1
**Saves to:** `checkpoints/curriculum/stage{N}_{name}.pt`

---

### Step 6.1 — Run Full Curriculum (All 6 Stages)

```powershell
python training\curriculum_training.py
```

**Total training:** 15,000 episodes across 6 stages

**Time estimate:**
- CPU: 8–16 hours
- GPU: 2–4 hours

---

### Step 6.2 — Run Specific Stages Only

**Start from Stage 3 (requires Stage 2 checkpoint):**
```powershell
python training\curriculum_training.py --start_stage 3 --end_stage 6
```

**Run only Stages 1 and 2 (quick test):**
```powershell
python training\curriculum_training.py --start_stage 1 --end_stage 2
```

**GPU curriculum with custom checkpoint directory:**
```powershell
python training\curriculum_training.py --device cuda --checkpoint_dir checkpoints\curriculum_v2
```

---

### Step 6.3 — Stage Checkpoint Files

After curriculum completes, you will have:
```
checkpoints/curriculum/
+-- stage1_single_nominal.pt       <- 1 sat, no eclipse
+-- stage2_single_eclipse.pt       <- 1 sat, eclipse
+-- stage3_three_nominal.pt        <- 3 sats, eclipse
+-- stage4_three_degradation.pt    <- 3 sats, degradation
+-- stage5_six_stress.pt           <- 6 sats, full stress
+-- stage6_twelve_adversarial.pt   <- 12 sats, anomaly injection
```

---

## 7. Running Simulations

Simulations run trained agents in the environment without gradient updates.
Use these to observe learned behaviour, generate plots, and debug.

---

### Step 7.1 — Quick Environment Smoke Test (No Training Needed)

```powershell
python -c "
import numpy as np
from environment.constellation_env import ConstellationEnv

env = ConstellationEnv(n_satellites=3, enable_eclipse=True)
obs, info = env.reset(seed=42)

labels = ['SoC', 'SoH', 'Temp', 'P_solar', 'Phase', 'Eclipse', 'P_cons', 'CommDly']
print('Initial state (3 satellites x 8 dimensions):')
for sat_id in range(3):
    vals = ', '.join(f'{l}={obs[sat_id,i]:.3f}' for i, l in enumerate(labels))
    print(f'  SAT-{sat_id}: {vals}')

print('\nRunning 20 steps with random actions...')
for step in range(20):
    actions = env.action_space.sample()
    obs, rewards, terminated, truncated, info = env.step(actions)
    print(f'  Step {step+1:2d}: actions={actions} | SoC={obs[:,0].round(3)} | rewards={[round(r,3) for r in rewards]}')
    if terminated or truncated:
        break

env.close()
print('Environment simulation OK.')
"
```

---

### Step 7.2 — Simulate with Trained MAPPO Agents

```powershell
python -c "
import numpy as np, torch
from environment.constellation_env import ConstellationEnv
from world_model.world_model import WorldModel
from agents.satellite_agent import SatelliteAgent
from agents.policy_network import ActorNetwork

n_sat = 3
env   = ConstellationEnv(n_satellites=n_sat, enable_eclipse=True)
wm    = WorldModel(state_dim=8, action_dim=5)
wm.load('checkpoints/world_model_best.pt')

actors = [ActorNetwork(obs_dim=8, future_dim=8, action_dim=5) for _ in range(n_sat)]
ckpt   = torch.load('checkpoints/mappo_best.pt', map_location='cpu')
for i, actor in enumerate(actors):
    key = f'actor_{i}'
    if key in ckpt:
        actor.load_state_dict(ckpt[key])
        print(f'Loaded actor {i}')

agents = [SatelliteAgent(agent_id=i, world_model=wm, actor=actors[i]) for i in range(n_sat)]

obs, _ = env.reset(seed=0)
total_reward, safety_overrides = 0.0, 0

print('Simulating 1 full orbit (552 steps at dt=10s)...')
for step in range(552):
    actions = []
    for i, agent in enumerate(agents):
        action, log_prob, value, info = agent.act(obs[i], deterministic=True)
        actions.append(action)
        safety_overrides += int(info['was_overridden'])
    obs, rewards, done, trunc, _ = env.step(np.array(actions))
    total_reward += sum(rewards)
    if step % 100 == 0:
        print(f'  Step {step:4d}: SoC={obs[:,0].round(3)} | actions={actions}')
    if done or trunc: break

print(f'Done | total_reward={total_reward:.3f} | safety_overrides={safety_overrides}')
env.close()
"
```

---

### Step 7.3 — Simulate with Algorithm 4 Explicit MPC

This uses `cognitive_decision()` — the explicit model-predictive control loop
that enumerates all 5 actions, scores each with world model predictions, and
picks the best. No trained actor needed.

```powershell
python -c "
import numpy as np
from environment.constellation_env import ConstellationEnv
from world_model.world_model import WorldModel
from agents.satellite_agent import SatelliteAgent
from agents.policy_network import ActorNetwork

env    = ConstellationEnv(n_satellites=3)
wm     = WorldModel(state_dim=8, action_dim=5)
wm.load('checkpoints/world_model_best.pt')
actors = [ActorNetwork(8, 8, 5) for _ in range(3)]
agents = [SatelliteAgent(agent_id=i, world_model=wm, actor=actors[i]) for i in range(3)]

obs, _ = env.reset(seed=7)
names  = {0:'payload_ON', 1:'payload_OFF', 2:'hibernate', 3:'relay_mode', 4:'charge_priority'}

for step in range(30):
    actions = []
    for i, agent in enumerate(agents):
        action, info = agent.cognitive_decision(obs[i], k=5)
        actions.append(action)
        if step == 0:
            print(f'SAT-{i} MPC action scores:')
            for a, score in info['scores'].items():
                print(f'  {names[a]:20s} = {score:.4f}')
            print(f'  -> Chosen: {names[action]} (override={info[\"was_overridden\"]})')
    obs, rewards, _, _, _ = env.step(np.array(actions))
    print(f'Step {step+1:2d}: {[names[a] for a in actions]} | SoC={obs[:,0].round(3)}')
env.close()
"
```

---

### Step 7.4 — Test Layer 4 Hierarchical Coordination

```powershell
python -c "
import numpy as np
from environment.constellation_env import ConstellationEnv
from world_model.world_model import WorldModel
from agents.satellite_agent import SatelliteAgent
from agents.policy_network import ActorNetwork
from coordination.cluster_coordinator import ClusterCoordinator
from marl.communication_protocol import CommunicationProtocol

n_sat = 3
env   = ConstellationEnv(n_satellites=n_sat)
wm    = WorldModel(state_dim=8, action_dim=5)
wm.load('checkpoints/world_model_best.pt')
actors = [ActorNetwork(8, 8, 5) for _ in range(n_sat)]
agents = [SatelliteAgent(agent_id=i, world_model=wm, actor=actors[i]) for i in range(n_sat)]

obs, _ = env.reset(seed=42)
coordinator = ClusterCoordinator(n_satellites=n_sat, forecast_horizon=10)
comm        = CommunicationProtocol(n_satellites=n_sat)

print('Running Layer 4 Coordination cycle...')
assignment = coordinator.coordinate(agents, comm, world_model=wm, current_obs=obs)
print('Task assignment from ClusterCoordinator:')
for sat_id, task in sorted(assignment.items()):
    print(f'  SAT-{sat_id}: {task}')
fleet = coordinator.assess_fleet_status(
    coordinator.aggregate([wm.predict_k_steps(obs[i], [1]*10, 10) for i in range(n_sat)])
)
print(f'Fleet health: {fleet}')
env.close()
"
```

---

### Step 7.5 — Test Layer 5 Safety System

```powershell
python -c "
import numpy as np
from safety.safety_monitor import SafetyMonitor, SafetyThresholds
from safety.anomaly_detector import AnomalyDetector
from safety.recovery_policy import RecoveryPolicy, AnomalyType

# Create monitors for satellite 0
monitor  = SafetyMonitor(sat_id=0)
detector = AnomalyDetector(sat_id=0)
recovery = RecoveryPolicy(sat_id=0)

# Simulate a low-SoC scenario
print('Testing Safety Monitor state machine...')
obs = np.array([0.25, 1.0, 0.3, 0.5, 0.0, 0.0, 0.25, 0.05], dtype='float32')  # SoC=0.25 (WARNING zone)
action, state, reason = monitor.check(obs, policy_action=0)  # 0 = payload_ON
print(f'SoC=0.25 (WARNING zone): state={state.value} | action={action} | reason={reason}')

obs[0] = 0.08  # SoC drops to 8% (CRITICAL)
action, state, reason = monitor.check(obs, policy_action=0)
print(f'SoC=0.08 (CRITICAL): state={state.value} | action={action} | reason={reason}')

# Trigger battery recovery
recovery.activate(AnomalyType.BATTERY_OVERDISCHARGE)
for phase_step in range(10):
    result = recovery.step(obs)
    print(f'  Recovery step {phase_step+1}: action={result.action} | phase={result.phase.name} | progress={result.progress:.2f}')
    if result.complete: break
"
```

---

## 8. Evaluating Trained Agents

### Step 8.1 — Run the Built-in Experiment Runner

```powershell
python evaluation\experiment_runner.py
```

---

### Step 8.2 — Manual Multi-Episode Evaluation

```powershell
python -c "
import numpy as np, torch
from environment.constellation_env import ConstellationEnv
from world_model.world_model import WorldModel
from agents.satellite_agent import SatelliteAgent
from agents.policy_network import ActorNetwork

N_EVAL = 20
n_sat  = 3

env    = ConstellationEnv(n_satellites=n_sat, enable_eclipse=True)
wm     = WorldModel(state_dim=8, action_dim=5)
wm.load('checkpoints/world_model_best.pt')
actors = [ActorNetwork(8, 8, 5) for _ in range(n_sat)]
ckpt   = torch.load('checkpoints/mappo_best.pt', map_location='cpu')
for i, a in enumerate(actors):
    if f'actor_{i}' in ckpt: a.load_state_dict(ckpt[f'actor_{i}'])
agents = [SatelliteAgent(agent_id=i, world_model=wm, actor=actors[i]) for i in range(n_sat)]

all_rewards, all_overrides, all_final_soc = [], [], []
for ep in range(N_EVAL):
    obs, _ = env.reset(seed=ep)
    ep_reward, overrides = 0.0, 0
    for _ in range(1000):
        actions = []
        for i, agent in enumerate(agents):
            action, _, _, info = agent.act(obs[i], deterministic=True)
            actions.append(action)
            overrides += int(info['was_overridden'])
        obs, rewards, done, trunc, _ = env.step(np.array(actions))
        ep_reward += sum(rewards)
        if done or trunc: break
    all_rewards.append(ep_reward)
    all_overrides.append(overrides)
    all_final_soc.append(obs[:,0].mean())

print(f'=== Evaluation Results ({N_EVAL} episodes) ===')
print(f'Mean Reward           : {np.mean(all_rewards):.3f} +/- {np.std(all_rewards):.3f}')
print(f'Max Reward            : {np.max(all_rewards):.3f}')
print(f'Min Reward            : {np.min(all_rewards):.3f}')
print(f'Mean Final SoC        : {np.mean(all_final_soc):.3f}')
print(f'Avg Safety Overrides  : {np.mean(all_overrides):.1f} per episode')
env.close()
"
```

---

## 9. Running Tests

### Step 9.1 — Run Full Test Suite

```powershell
pytest tests\ -v
```

### Step 9.2 — Run Tests for Specific Layers

```powershell
pytest tests\test_environment.py -v      # Layer 1 physics
pytest tests\test_agents.py -v           # Layer 2 agents
pytest tests\test_marl.py -v             # Layer 3 MARL
pytest tests\test_safety.py -v           # Layer 5 safety
```

### Step 9.3 — Coverage Report

```powershell
pytest tests\ --cov=. --cov-report=term-missing -v
```

### Step 9.4 — Quick Sanity Check (Single Command)

```powershell
python -m pytest tests\ -q --tb=short
```

---

## 10. Monitoring Training

### Step 10.1 — Launch TensorBoard (While Training Is Running)

Open a second PowerShell terminal and run:

```powershell
.venv\Scripts\Activate.ps1
tensorboard --logdir runs\
```

Then open your browser at: **http://localhost:6006**

**What each metric means:**

| Metric | Description | Healthy Trend |
|---|---|---|
| `episode_reward` | Sum of cooperative rewards per episode | Increasing |
| `actor_loss` | PPO surrogate objective per agent | Decreasing or stable |
| `critic_loss` | Value function MSE loss | Decreasing |
| `entropy` | Policy distribution entropy | Slowly decreasing |
| `safety_override_rate` | Fraction of steps where safety gate fires | Low and decreasing |
| `mean_soc` | Mean battery SoC across satellites | Above 0.3 |

---

### Step 10.2 — Save Training Output to File

```powershell
python training\train_marl.py 2>&1 | Tee-Object training_log.txt
```

---

## 11. Configuration Reference

### `config/training.yaml` — Hyperparameters

| Parameter | Default | What It Controls |
|---|---|---|
| `n_episodes` | 10000 | Total training episodes |
| `episode_length` | 1000 | Steps per episode (1000 x 10s = ~2.8 orbits) |
| `lr_actor` | 3e-4 | Actor learning rate (too high = unstable, too low = slow) |
| `lr_critic` | 1e-3 | Critic LR (higher than actor is standard in PPO) |
| `gamma` | 0.99 | Discount factor (higher = more long-term focus) |
| `clip_epsilon` | 0.2 | PPO clip range (0.1–0.3 typical) |
| `entropy_coef` | 0.01 | Entropy bonus (higher = more exploration) |
| `rollout_length` | 200 | Steps collected between PPO updates |
| `n_epochs_per_update` | 10 | PPO update epochs per rollout |
| `device` | auto | cpu, cuda, or auto |

### `config/environment.yaml` — Physics Parameters

| Parameter | Default | What It Controls |
|---|---|---|
| `n_satellites` | 3 | Number of satellites in the constellation |
| `battery.capacity_wh` | 100 | Battery energy capacity (larger = harder to discharge) |
| `battery.soc_min` | 0.15 | Minimum safe SoC, below this triggers CRITICAL state |
| `thermal.T_max` | 60 C | Thermal safety limit, above this triggers thermal recovery |
| `solar.efficiency` | 0.28 | Solar cell efficiency (28% = triple-junction cells) |
| `orbital.dt` | 10 s | Simulation timestep (smaller = more accurate, slower) |

### Overriding Config from Command Line

All training scripts accept CLI flags that override YAML defaults:

```powershell
python training\train_marl.py --n_episodes 3000 --lr_actor 1e-4 --n_satellites 6
```

---

## 12. Troubleshooting

### ERROR: `ModuleNotFoundError: No module named 'environment'`

**Cause:** Package not installed in editable mode, or venv not activated.

**Fix:**
```powershell
.venv\Scripts\Activate.ps1
pip install -e .
```

---

### ERROR: `CUDA out of memory`

**Cause:** Batch size too large for GPU memory.

**Fix:** Reduce batch size and rollout length:
```powershell
python training\train_marl.py --batch_size 128 --rollout_length 100
```

---

### ERROR: `SyntaxError` on `list[int]` or `dict[str, Any]`

**Cause:** Python 3.9 or earlier.

**Fix:** Install Python 3.10 or 3.11 from https://python.org, recreate the venv.

---

### ERROR: World model checkpoint not found

**Cause:** MARL training started before world model training completed.

**Fix:** Always run phases in order:
```powershell
python training\train_world_model.py   # Phase 1 first
python training\train_marl.py          # Phase 2 second
```

---

### Training reward not improving after 1000+ episodes

| Cause | Fix |
|---|---|
| World model untrained | Run `train_world_model.py` first, verify predictions |
| Learning rate too high | Try `--lr_actor 1e-4 --lr_critic 5e-4` |
| Entropy collapsed | Try `--entropy_coef 0.05` for more exploration |
| Episode too short | Try `--episode_length 2000` |
| Too few rollout steps | Try `--rollout_length 400` |

---

### ERROR: `scipy.optimize.milp not found`

**Cause:** scipy < 1.7 installed.

**Fix:**
```powershell
pip install "scipy>=1.11.0"
```

---

### PowerShell script execution blocked

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 13. Quick Reference Commands

```powershell
# ===== ONE-TIME SETUP =====
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

# ===== VERIFY =====
python -c "from environment.constellation_env import ConstellationEnv; print('OK')"

# ===== PHASE 1: Train World Model =====
python training\train_world_model.py
# With GPU:
python training\train_world_model.py --device cuda

# ===== PHASE 2: Train MARL Agents =====
python training\train_marl.py
# With GPU, 6 satellites:
python training\train_marl.py --device cuda --n_satellites 6 --n_episodes 8000

# ===== PHASE 3: Full Curriculum =====
python training\curriculum_training.py
# Resume from Stage 3:
python training\curriculum_training.py --start_stage 3

# ===== SIMULATE =====
# Quick env test (no checkpoint needed):
python -c "from environment.constellation_env import ConstellationEnv; env=ConstellationEnv(3); obs,_=env.reset(); [env.step(env.action_space.sample()) for _ in range(100)]; print('OK'); env.close()"

# ===== EVALUATE =====
python evaluation\experiment_runner.py

# ===== TESTS =====
pytest tests\ -v

# ===== TENSORBOARD (in separate terminal) =====
tensorboard --logdir runs\
# Open: http://localhost:6006
```
