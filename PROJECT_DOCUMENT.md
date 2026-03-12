# CASC-RL: Cognitive Autonomous Satellite Constellation with Reinforcement Learning
### Complete Project Document — From Scratch to Production

---

> **Project Type:** Research-Grade AI/ML System  
> **Domain:** Aerospace Systems · Multi-Agent Reinforcement Learning · Digital Twin  
> **Target Outcome:** Publication-ready experimental framework with full constellation simulation  
> **Date:** March 2026

---

## Table of Contents

1. [Project Vision & Abstract](#1-project-vision--abstract)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Layered Architecture Deep Dive](#3-layered-architecture-deep-dive)
4. [Core Algorithms](#4-core-algorithms)
5. [End-to-End System Workflow](#5-end-to-end-system-workflow)
6. [Implementation Phases](#6-implementation-phases)
7. [Repository Structure](#7-repository-structure)
8. [Technology Stack & Dependencies](#8-technology-stack--dependencies)
9. [Configuration System](#9-configuration-system)
10. [Training Pipeline](#10-training-pipeline)
11. [Evaluation & Benchmarking](#11-evaluation--benchmarking)
12. [Experiments & Paper Outputs](#12-experiments--paper-outputs)
13. [Safety & Fault Recovery](#13-safety--fault-recovery)
14. [Visualization & Dashboard](#14-visualization--dashboard)
15. [Production Deployment](#15-production-deployment)
16. [Testing Strategy](#16-testing-strategy)
17. [Contribution Guide & Research Extensions](#17-contribution-guide--research-extensions)

---

## 1. Project Vision & Abstract

### 1.1 What is CASC-RL?

**CASC-RL** (Cognitive Autonomous Satellite Constellation — Reinforcement Learning) is a research-grade framework that combines:

- **Physics-accurate Digital Twin simulation** of satellite orbital and hardware dynamics
- **Predictive World Models** that allow each satellite agent to forecast future system states
- **Multi-Agent Reinforcement Learning (MARL)** using MAPPO for cooperative constellation-level decision-making
- **Hierarchical Coordination** for global task allocation, scheduling, and resource optimization
- **Autonomous Safety Recovery** for anomaly detection, diagnosis, and fault response

### 1.2 Why This Matters

Traditional satellite autonomy relies on rule-based or PID controllers that cannot adapt. Modern satellite constellations (e.g., Starlink, OneWeb) operate with thousands of interdependent nodes — requiring intelligent, self-organizing control. CASC-RL is among the first frameworks to integrate **all five pillars** into a unified learning system:

| Pillar | Method Used |
|---|---|
| Physics Simulation | Orbital mechanics, thermal models, battery degradation |
| Predictive Modeling | Neural World Model (fψ) |
| Agent Learning | PPO-based local RL policy |
| Cooperative Learning | MAPPO with centralized critic |
| Hierarchical Planning | Cluster Coordinator + Task Allocator |

### 1.3 Research Objectives

1. Demonstrate superior battery lifetime management vs. rule-based and PID baselines
2. Enable cooperative task allocation without centralized ground control
3. Predict and pre-empt eclipse/power failures using model-based foresight
4. Scale to constellation sizes of 3, 6, and 12 satellites
5. Produce publication-ready figures and metrics

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CASC-RL System Architecture                      │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │      Layer 4: Hierarchical Coordination Layer               │   │
│  │  [Cluster Coordinator] [Task Allocator] [Payload Scheduler] │   │
│  └─────────────────────────┬───────────────────────────────────┘   │
│                             │ Mission Commands                      │
│  ┌─────────────────────────▼───────────────────────────────────┐   │
│  │      Layer 3: Cooperative Intelligence Layer                │   │
│  │  [MAPPO Policy] [Communication Module] [Reward Shaping]     │   │
│  └─────────────────────────┬───────────────────────────────────┘   │
│                             │ Cooperative Actions                   │
│  ┌─────────────────────────▼───────────────────────────────────┐   │
│  │      Layer 2: Satellite Cognitive Layer (per agent)         │   │
│  │  [World Model (fψ)] [Local RL Policy] [Safety Monitor]      │   │
│  └─────────────────────────┬───────────────────────────────────┘   │
│                             │ State Observations                    │
│  ┌─────────────────────────▼───────────────────────────────────┐   │
│  │      Layer 1: Physical Environment & Digital Twin           │   │
│  │  [Orbital Dynamics] [Eclipse] [Solar] [Battery] [Thermal]   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1 Algorithm-to-Layer Mapping

| Layer | Function | Algorithms |
|---|---|---|
| L1: Physical | Environment, Dynamics | Orbital, Power, Thermal Models |
| L2: Cognitive (Local) | Prediction, Decision | World Model, RL Policy, Safety Gate |
| L3: Cooperative | Multi-Agent Learning | MAPPO, Reward Shaping, Comm. |
| L4: Coordination | Task Allocation | Cluster Agent, Forecasting, Scheduling |
| L5: Recovery | Anomaly Handling | Detection, Diagnosis, Recovery |

---

## 3. Layered Architecture Deep Dive

### Layer 1 — Physical Environment & Digital Twin

**Purpose:** Simulate space dynamics and all hardware behavior with high physical fidelity.

**Responsibilities:**

| Module | Function |
|---|---|
| `orbital_dynamics.py` | Keplerian orbit propagation, position/velocity computation |
| `eclipse_model.py` | Eclipse detection based on orbital geometry and umbra/penumbra |
| `solar_model.py` | Solar irradiance × panel area × efficiency → solar power |
| `battery_model.py` | SoC update via charge/discharge dynamics (Coulomb counting + Peukert) |
| `degradation_model.py` | SoH decay with charge cycle count and temperature |
| `thermal_model.py` | Nodal thermal model: eclipse/sunlit heating, radiation, dissipation |
| `constellation_env.py` | Top-level Gym environment combining all modules |

**State Vector Output:**

```
s_t = [SoC, SoH, temperature, solar_input, orbital_phase,
       eclipse_flag, power_consumption, comm_delay]
```

---

### Layer 2 — Satellite Cognitive Layer

**Purpose:** Each satellite acts as an intelligent autonomous agent with memory and foresight.

**Modules:**

| Module | Function |
|---|---|
| `world_model.py` | Neural network fψ: predicts ŝ_{t+k} from (s_t, a_{t:t+k-1}) |
| `policy_network.py` | Actor network: π(a | s_t, ŝ_{t+k}) |
| `critic_network.py` | Value estimator V(s) for advantage computation |
| `safety_monitor.py` | Constraint checking: SoC > SoC_min, temp < T_max |
| `action_selector.py` | Merge policy output with safety constraints |

**Actions Available to Each Agent:**

```
A = {payload_ON, payload_OFF, hibernate, relay_mode, charge_priority}
```

---

### Layer 3 — Cooperative Intelligence Layer

**Purpose:** Multi-agent coordination via shared learning and communication.

**Modules:**

| Module | Function |
|---|---|
| `mappo_trainer.py` | MAPPO training loop with centralized critic |
| `cooperative_rewards.py` | Global + local reward blending |
| `buffer.py` | Replay buffer: stores (s, a, r, s', done) |
| `advantage_estimator.py` | GAE (Generalized Advantage Estimation) |
| `communication_protocol.py` | Inter-satellite message passing (compressed state broadcasts) |

---

### Layer 4 — Hierarchical Coordination Layer

**Purpose:** Constellation-level strategy and global resource planning.

**Modules:**

| Module | Function |
|---|---|
| `cluster_coordinator.py` | Aggregates predicted states from all agents |
| `task_allocator.py` | Solves constrained assignment problem (ILP/greedy) |
| `scheduling.py` | Temporal mission plan: when each sat performs what task |
| `communication_protocol.py` | Broadcast mission commands to agents |

**Example Output:**

```
Satellite 1 → Relay mode
Satellite 2 → Hibernate
Satellite 3 → Payload active
```

---

### Layer 5 — Safety & Recovery (Implicit Layer)

**Purpose:** Fault detection, safe fallback, and autonomous recovery.

| Module | Function |
|---|---|
| `anomaly_detector.py` | Statistical or ML-based anomaly detection |
| `safety_monitor.py` | Hard constraint enforcement |
| `recovery_policy.py` | Pre-defined safe recovery sequences |

---

## 4. Core Algorithms

### Algorithm 1 — Satellite Environment Simulation

**Purpose:** Simulate orbital dynamics and system physics.

```python
# Pseudocode
initialize constellation_env(n_satellites, orbital_params)

for t in range(T_max):
    for i in range(n_satellites):
        pos_i, vel_i      = update_orbital_position(t, orbital_params[i])
        eclipse_i         = detect_eclipse(pos_i, sun_position)
        solar_power_i     = compute_solar_power(eclipse_i, panel_area, efficiency)
        P_consumed_i      = compute_power_consumption(mode_i)
        SoC_i             = update_battery(SoC_i, solar_power_i, P_consumed_i, dt)
        T_i               = update_temperature(T_i, eclipse_i, P_consumed_i, dt)
        SoH_i             = update_degradation(SoH_i, SoC_i, T_i, cycle_count_i)
    
    s_t = [SoC, SoH, T, solar_input, orbital_phase, eclipse_flag]
    yield s_t
```

**Output:** `s_t = [SoC, SoH, temperature, solar_input, orbital_phase]`

---

### Algorithm 2 — World Model Learning

**Purpose:** Train a neural network to predict future satellite states.

**Model:** `ŝ_{t+k} = fψ(s_t, a_{t:t+k-1})`

```python
# Pseudocode
dataset = collect_transition_samples(env, random_policy, N=100000)
# dataset contains: (s_t, a_t, s_{t+1}) tuples

model = DynamicsNetwork(input_dim=state_dim + action_dim,
                        output_dim=state_dim,
                        hidden=[256, 256])

for epoch in range(N_epochs):
    s, a, s_next = sample_batch(dataset)
    s_pred = model(s, a)
    loss = MSE(s_pred, s_next)
    optimizer.step(loss)

# Multi-step prediction via rollout
def predict_k_steps(s_t, actions, k):
    s = s_t
    for i in range(k):
        s = model(s, actions[i])
    return s  # ŝ_{t+k}
```

**Predictions:** battery SoC, temperature, eclipse timing, degradation trend

---

### Algorithm 3 — Cooperative MARL Training (MAPPO)

**Purpose:** Train constellation-level cooperative policies.

**Objective:**

```
L_PPO = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]

where r_t = π_θ(a|s) / π_θ_old(a|s)   (probability ratio)
      A_t = GAE advantage estimate
      ε   = 0.2 (clip parameter)
```

```python
# Pseudocode
policies = [ActorNetwork() for i in range(n_satellites)]
critic   = CentralizedCritic(global_state_dim)

for episode in range(N_episodes):
    trajectories = collect_trajectories(env, policies, T=200)
    
    advantages = compute_GAE(trajectories, critic, gamma=0.99, lambda=0.95)
    
    for epoch in range(K_epochs):
        for batch in trajectories:
            # Policy update (PPO clip)
            ratio = new_policy(a|s) / old_policy(a|s)
            L_clip = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
            policy_loss = -mean(L_clip)
            
            # Value update
            value_loss = MSE(critic(global_s), returns)
            
            # Gradient sharing across agents
            share_gradients(policies)
            optimizer.step(policy_loss + value_loss)
```

---

### Algorithm 4 — Cognitive Decision Making

**Purpose:** Select actions using both current state and model-predicted future states.

```python
# Pseudocode
def cognitive_decision(s_t, world_model, policy, k=5):
    # Generate k-step predictions for each candidate action
    candidate_actions = [PAYLOAD_ON, HIBERNATE, RELAY_MODE]
    
    best_action, best_score = None, -inf
    
    for a_candidate in candidate_actions:
        # Predict future under this action
        action_seq = [a_candidate] * k
        s_future = world_model.predict_k_steps(s_t, action_seq, k)
        
        # Compute expected reward
        r = w1 * s_future.SoC \
          - w2 * s_future.degradation_rate \
          - w3 * s_future.thermal_risk
        
        if r > best_score:
            best_action = a_candidate
            best_score  = r
    
    # Apply safety filter
    safe_action = safety_monitor.filter(best_action, s_t)
    return safe_action
```

**Reward Weights:** `w1=1.0 (SoC), w2=0.5 (Degradation), w3=0.3 (Thermal Risk)`

---

### Algorithm 5 — Hierarchical Coordination

**Purpose:** Coordinate all satellites at constellation level.

```python
# Pseudocode
def hierarchical_coordination(satellite_agents, world_model, coordinator):
    # Step 1: Collect predicted state forecasts from each satellite
    forecasts = [agent.world_model.predict(horizon=10) for agent in satellite_agents]
    
    # Step 2: Cluster coordinator aggregates
    global_forecast = coordinator.aggregate(forecasts)
    
    # Step 3: Solve constrained task allocation
    # maximize: sum(mission_reward_i)
    # subject to:
    #   sum(power_i) <= P_total
    #   comm_links satisfy latency constraints
    
    assignment = coordinator.solve_allocation(
        global_forecast,
        power_budget=P_total,
        comm_constraints=comm_graph
    )
    
    # Step 4: Broadcast mission commands
    for sat_id, task in assignment.items():
        satellite_agents[sat_id].receive_command(task)
    
    return assignment
```

---

## 5. End-to-End System Workflow

```
┌──────────────────────────────────┐
│   Digital Twin Environment       │  ← orbital_dynamics, solar, battery, thermal
│   (constellation_env.py)         │
└──────────────┬───────────────────┘
               │ s_t
               ▼
┌──────────────────────────────────┐
│   State Observation              │  ← sensor readings per satellite
└──────────────┬───────────────────┘
               │ s_t
               ▼
┌──────────────────────────────────┐
│   World Model Prediction         │  ← fψ: (s_t, a) → ŝ_{t+k}
│   (world_model.py)               │
└──────────────┬───────────────────┘
               │ ŝ_{t+k}
               ▼
┌──────────────────────────────────┐
│   Local RL Policy Decision       │  ← cognitive action selection
│   (satellite_agent.py)           │
└──────────────┬───────────────────┘
               │ local actions
               ▼
┌──────────────────────────────────┐
│   Cooperative MARL Layer         │  ← MAPPO + shared rewards
│   (mappo_trainer.py)             │
└──────────────┬───────────────────┘
               │ cooperative actions
               ▼
┌──────────────────────────────────┐
│   Hierarchical Coordinator       │  ← task allocation + scheduling
│   (cluster_coordinator.py)       │
└──────────────┬───────────────────┘
               │ mission commands
               ▼
┌──────────────────────────────────┐
│   Satellite Actions Executed     │  ← payload ON/hibernate/relay/charge
└──────────────┬───────────────────┘
               │ applied actions
               ▼
┌──────────────────────────────────┐
│   Environment Update             │  ← physics simulation advances by Δt
└──────────────┬───────────────────┘
               │
               └──────────────────── Repeat ◄─────────────────────────
```

---

## 6. Implementation Phases

### Phase 0: Project Setup & Environment (Week 1-2)

**Goal:** Establish the project scaffold, tooling, and dependencies.

- [ ] Initialize git repository and project structure
- [ ] Create `requirements.txt` with all dependencies
- [ ] Set up `config/` YAML files for environment, training, constellation
- [ ] Configure logging (WandB or TensorBoard)
- [ ] Write unit test scaffold (`tests/` directory)
- [ ] Set up CI pipeline (GitHub Actions)

**Deliverables:**
- Working repository with installable package
- All config files verified to load correctly

---

### Phase 1: Physical Environment / Digital Twin (Week 2-4)

**Goal:** Implement the Layer 1 physics simulation.

#### 1.1 Orbital Dynamics (`orbital_dynamics.py`)
- Implement Keplerian orbit propagator (two-body problem)
- Add J2 perturbation for higher accuracy
- Compute satellite position and velocity over one orbital period (~90 min LEO)

```python
class OrbitalDynamics:
    def __init__(self, semi_major_axis, eccentricity, inclination, ...):
        ...
    def propagate(self, t) -> (position, velocity):
        ...
```

#### 1.2 Eclipse Detection (`eclipse_model.py`)
- Compute sun vector in ECI frame
- Detect eclipse entry/exit based on Earth shadow geometry
- Output: binary flag + eclipse fraction

#### 1.3 Solar Power Generation (`solar_model.py`)
- `P_solar = η × A_panel × G_solar × cos(θ) × (1 - eclipse_flag)`
- Include solar flux variation with distance

#### 1.4 Battery Model (`battery_model.py`)
- Coulomb counting: `SoC(t+1) = SoC(t) + (P_solar - P_consumed) × dt / C_bat`
- Include charge/discharge efficiency terms

#### 1.5 Battery Degradation (`degradation_model.py`)
- Wöhler-based cycle degradation: `SoH -= α × ΔDoD^β`
- Temperature-accelerated degradation via Arrhenius model

#### 1.6 Thermal Model (`thermal_model.py`)
- Nodal heat balance: `m·c·dT/dt = Q_solar + Q_internal - Q_radiated`
- Eclipse vs. sunlit thermal profiles

#### 1.7 Constellation Environment (`constellation_env.py`)
- Implement OpenAI Gym-compatible interface
- `reset()`, `step(actions)`, `observation_space`, `action_space`
- Vectorized environment for multi-satellite support

**Deliverables:**
- All physics modules with unit tests
- Single-satellite orbit simulation demo
- Gym environment that runs a full 24-hour simulation

---

### Phase 2: World Model Training (Week 4-6)

**Goal:** Train a neural network to predict future satellite states.

#### 2.1 Dataset Generation (`dataset_builder.py`)
- Run physics environment with random actions
- Collect N=100,000 transitions `(s_t, a_t, s_{t+1})`
- Save in HDF5 or npz format

#### 2.2 Dynamics Network Architecture (`dynamics_network.py`)
- MLP or LSTM architecture (LSTM for temporal dependencies)
- Multi-step prediction: unroll k=5-10 steps
- Include uncertainty estimation (ensemble or Bayesian)

```python
class DynamicsNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.encoder = MLP([state_dim + action_dim, 256, 256])
        self.decoder = MLP([256, state_dim])
    
    def forward(self, s, a) -> s_next_pred:
        ...
```

#### 2.3 World Model Training (`training.py`)
- Train with MSE loss: `L = ||s_{t+1} - ŝ_{t+1}||²`
- Validate on held-out test trajectories
- Log prediction accuracy per state dimension (SoC, T, SoH)

**Deliverables:**
- Trained world model with <5% prediction error per dimension
- Prediction accuracy plots for battery SoC, temperature, eclipse timing

---

### Phase 3: Individual Satellite RL Agent (Week 6-9)

**Goal:** Train each satellite's local RL policy with cognitive world-model integration.

#### 3.1 Policy Network (`policy_network.py`)
- Actor: `s_t + ŝ_{t+k} → a` (uses world model predictions as additional input)
- Discrete action head with softmax or continuous action head
- Network: 3-layer MLP with LayerNorm

```python
class ActorNetwork(nn.Module):
    def __init__(self):
        self.net = MLP([state_dim + pred_dim, 256, 256, action_dim])
    
    def forward(self, s_t, s_future) -> action_distribution:
        ...
```

#### 3.2 Critic Network (`critic_network.py`)
- Value estimator V(s): `state → scalar value`
- Includes global state in MARL training

#### 3.3 Reward Function Design
```python
def compute_reward(s_t, a_t, s_next):
    r_soc       = w1 * s_next.SoC                     # maximize SoC
    r_degrade   = -w2 * (s_t.SoH - s_next.SoH)        # penalize SoH loss  
    r_thermal   = -w3 * max(0, s_next.temp - T_max)   # thermal penalty
    r_mission   = w4 * mission_completion(a_t)         # reward task execution
    return r_soc + r_degrade + r_thermal + r_mission
```

#### 3.4 Action Selector with Safety Filter (`action_selector.py`)
- Check: `SoC > SoC_min_threshold` before allowing payload_ON
- Check: `temperature < T_max` before disabling thermal regulation

#### 3.5 Single-Agent PPO Training (`train_marl.py`)
- Validate convergence on single satellite before MARL

**Deliverables:**
- Converged single-satellite agent
- Learning curves (reward vs. episode)
- Compared vs. rule-based baseline

---

### Phase 4: Cooperative MARL Training (Week 9-12)

**Goal:** Implement and train the full multi-agent MAPPO system.

#### 4.1 MAPPO Trainer (`mappo_trainer.py`)
- Centralized training: shared critic observes global state
- Decentralized execution: each agent uses only local observations
- Gradient clipping, entropy regularization, learning rate scheduling

```python
class MAPPOTrainer:
    def __init__(self, n_agents, actor_nets, critic_net):
        ...
    
    def train_episode(self, env):
        trajectories = self.rollout(env)
        advantages   = self.gae(trajectories)
        self.update_policies(trajectories, advantages)
        self.update_critic(trajectories)
```

#### 4.2 Shared Experience Buffer (`buffer.py`)
- Stores: `(s_local_i, s_global, a_i, r_i, done)` for all agents
- Configurable buffer size (minibatch sampling)

#### 4.3 Cooperative Reward Shaping (`cooperative_rewards.py`)
- Global team reward: covers constellation-level mission success
- Local bonus: individual SoC and thermal safety
- Conflict penalty: penalize two satellites competing for same relay slot

```python
def cooperative_reward(local_rewards, global_outcome, conflict_detected):
    return α * mean(local_rewards) + β * global_outcome - γ * conflict_detected
```

#### 4.4 Advantage Estimation (`advantage_estimator.py`)
- Generalized Advantage Estimation (GAE): `A_t = Σ (γλ)^l δ_{t+l}`

**Deliverables:**
- MARL system training on 3-satellite constellation
- Cooperative learning curves
- Comparison: MARL vs. independent RL vs. rule-based

---

### Phase 5: Hierarchical Coordination Layer (Week 12-15)

**Goal:** Implement constellation-level coordination above the MARL layer.

#### 5.1 Cluster Coordinator (`cluster_coordinator.py`)
- Receives predicted states from all agents
- Maintains global state estimate
- Determines which missions are viable given resource forecasts

#### 5.2 Task Allocator (`task_allocator.py`)
- Greedy or ILP-based assignment
- Objective: `max Σ mission_value(i) × x_i`
- Constraints: power budget, communication windows, thermal limits

```python
def solve_allocation(forecasts, power_budget, comm_constraints):
    # Linear Programming or Hungarian algorithm
    ...
    return {satellite_id: assigned_task}
```

#### 5.3 Payload Scheduler (`scheduling.py`)
- Temporal schedule over upcoming orbital period
- Handles communication windows (ISL: Inter-Satellite Links)
- Priority queue for urgent tasks (emergency relay vs. science payload)

#### 5.4 Communication Protocol (`communication_protocol.py`)
- Model communication delay based on inter-satellite distance
- Local broadcast vs. ground relay communication modes

**Deliverables:**
- Full hierarchical system tested on 12-satellite constellation
- Task allocation demo: correct assignment with power constraints
- Schedule visualization for one full orbit

---

### Phase 6: Safety & Fault Recovery Layer (Week 15-17)

**Goal:** Add robust safety monitoring and autonomous recovery.

#### 6.1 Anomaly Detector (`anomaly_detector.py`)
- Statistical threshold detection (Z-score on sensor readings)
- ML-based prediction of anomaly risk using world model residuals
- Detection of: battery over-discharge, overcurrent, thermal runaway

#### 6.2 Safety Monitor (`safety_monitor.py`)
- Hard constraint enforcement (overrides RL policy if violated)
- State machine: `NOMINAL → WARNING → CRITICAL → RECOVERY`

```python
class SafetyMonitor:
    def check(self, state) -> status:
        if state.SoC < SoC_CRITICAL:
            return EMERGENCY_HIBERNATE
        if state.temperature > T_CRITICAL:
            return THERMAL_SAFE_MODE
        return NOMINAL
```

#### 6.3 Recovery Policy (`recovery_policy.py`)
- Pre-programmed recovery sequences (not RL, hardcoded safe behaviors)
- Graceful degradation: partial hibernation, reduced payload, relay-only mode
- Autonomous re-entry to normal operation after recovery

**Deliverables:**
- Safety system handles eclipse stress test without SoC hitting 0
- Tested on 24-hour simulated anomaly injection scenarios

---

### Phase 7: Training Pipeline Integration (Week 17-19)

**Goal:** Integrate all training scripts into a reproducible pipeline.

```
train_world_model.py     → Pretrain fψ on random-policy data
        ↓
train_marl.py            → Train MAPPO with world model
        ↓
curriculum_training.py   → Progressive difficulty (1 sat → 3 → 6 → 12)
        ↓
evaluate_agent.py        → Benchmark against baselines
        ↓
experiment_runner.py     → Run ablation and stress experiments
        ↓
generate_plots.py        → Produce paper figures
```

#### Curriculum Training Strategy (`curriculum_training.py`)
- Stage 1: Single satellite, nominal conditions (Eclipse-free)
- Stage 2: Single satellite, full eclipse cycles
- Stage 3: 3-satellite constellation, nominal
- Stage 4: 3-satellite constellation + degradation
- Stage 5: 6-satellite constellation, full stress conditions
- Stage 6: 12-satellite constellation, adversarial conditions

**Deliverables:**
- Full training pipeline runnable with single command
- Curriculum logs show stable improvement at each stage

---

### Phase 8: Evaluation & Benchmarking (Week 19-21)

**Goal:** Rigorous comparison against baselines.

#### Baselines

| Baseline | Description | File |
|---|---|---|
| PID Rule Controller | Fixed `charge if SoC < 30%` strategy | `baseline_pid.py` |
| Rule-Based Scheduler | Heuristic mission scheduler | `baseline_rule.py` |
| Independent PPO | Non-cooperative agents | (ablation) |
| CASC-RL (Ours) | Full system | — |

#### Metrics (`metrics.py`)

| Metric | Description |
|---|---|
| Battery Lifetime | Episodes before SoH < 70% |
| Mission Success Rate | % of tasks completed per orbit |
| SoC Min Value | Worst-case battery state |
| Thermal Violations | Number of T > T_max events |
| Coordination Efficiency | Task allocation optimality |

**Deliverables:**
- Comparison table across all baselines
- Statistical significance (5 seeds per experiment)

---

### Phase 9: Visualization & Dashboard (Week 21-23)

**Goal:** Produce publication-quality figures and interactive monitoring.

#### 9.1 Constellation View (`constellation_view.py`)
- 3D orbital plot: satellite positions over time
- Color-coded by current mode (payload / hibernate / relay)
- Eclipse zone visualization

#### 9.2 Prediction Plots (`prediction_plots.py`)
- World model predicted vs. actual: SoC, temperature
- Prediction error over multi-step horizon
- Eclipse timing accuracy

#### 9.3 Interactive Dashboard (`dashboard.py`)
- Real-time constellation status (powered by Plotly Dash or Streamlit)
- Live charts: SoC per satellite, thermal, reward
- Mission control panel: inject anomalies, change task priorities

**Deliverables:**
- All paper figures (see Section 12)
- Runnable dashboard server

---

### Phase 10: Production & Deployment (Week 23-26)

**Goal:** Package and harden the system for production use.

- [ ] Containerize with Docker (`Dockerfile`, `docker-compose.yml`)
- [ ] REST API for ground station integration (`api/ground_station.py`)
- [ ] Model serialization and versioning (ONNX or TorchScript)
- [ ] Logging and telemetry pipeline
- [ ] Configuration management for different constellation sizes
- [ ] Documentation site (MkDocs or Sphinx)
- [ ] Performance profiling and GPU optimization

---

## 7. Repository Structure

```
casc-rl/
│
├── README.md                        # Project overview and quickstart
├── PROJECT_DOCUMENT.md              # This file
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup
├── Makefile                         # Common commands (train, test, eval)
│
├── config/
│   ├── environment.yaml             # Physics simulation params
│   ├── training.yaml                # RL training hyperparameters
│   └── constellation.yaml          # Satellite counts, orbital params
│
├── environment/                     # Layer 1: Physical Simulation
│   ├── __init__.py
│   ├── orbital_dynamics.py
│   ├── eclipse_model.py
│   ├── solar_model.py
│   ├── battery_model.py
│   ├── degradation_model.py
│   ├── thermal_model.py
│   └── constellation_env.py
│
├── world_model/                     # Layer 2a: Predictive World Model
│   ├── __init__.py
│   ├── world_model.py
│   ├── dynamics_network.py
│   ├── training.py
│   └── dataset_builder.py
│
├── agents/                          # Layer 2b: Satellite Agent
│   ├── __init__.py
│   ├── satellite_agent.py
│   ├── policy_network.py
│   ├── critic_network.py
│   └── action_selector.py
│
├── marl/                            # Layer 3: Cooperative MARL
│   ├── __init__.py
│   ├── mappo_trainer.py
│   ├── buffer.py
│   ├── advantage_estimator.py
│   └── cooperative_rewards.py
│
├── coordination/                    # Layer 4: Hierarchical Coordination
│   ├── __init__.py
│   ├── cluster_coordinator.py
│   ├── task_allocator.py
│   ├── scheduling.py
│   └── communication_protocol.py
│
├── safety/                          # Layer 5: Safety & Recovery
│   ├── __init__.py
│   ├── anomaly_detector.py
│   ├── safety_monitor.py
│   └── recovery_policy.py
│
├── training/                        # Training Pipeline
│   ├── train_world_model.py
│   ├── train_marl.py
│   └── curriculum_training.py
│
├── evaluation/                      # Benchmarks & Metrics
│   ├── baseline_pid.py
│   ├── baseline_rule.py
│   ├── experiment_runner.py
│   └── metrics.py
│
├── visualization/                   # Figures & Dashboard
│   ├── constellation_view.py
│   ├── prediction_plots.py
│   └── dashboard.py
│
├── experiments/                     # Experimental Scenarios
│   ├── nominal_conditions.py
│   ├── eclipse_stress_test.py
│   ├── degradation_test.py
│   └── constellation_scaling.py
│
├── tests/                           # Unit Tests
│   ├── test_environment.py
│   ├── test_world_model.py
│   ├── test_agents.py
│   └── test_marl.py
│
└── docs/                            # Documentation
    ├── architecture.md
    ├── algorithms.md
    └── api_reference.md
```

---

## 8. Technology Stack & Dependencies

```txt
# requirements.txt

# Core RL & ML
torch>=2.0.0
torchvision>=0.15.0
gymnasium>=0.29.0
stable-baselines3>=2.0.0

# Scientific Computing
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.0.0

# Orbital Mechanics
poliastro>=0.17.0     # Keplerian orbit propagation
astropy>=5.3.0        # Coordinate transforms, time systems

# Experiment Tracking
wandb>=0.15.0
tensorboard>=2.14.0

# Optimization
cvxpy>=1.4.0          # For ILP task allocation
ortools>=9.7.0        # Google OR-Tools (alternative solver)

# Visualization
matplotlib>=3.7.0
plotly>=5.15.0
dash>=2.13.0          # Interactive dashboard

# Config & Utilities
pyyaml>=6.0.1
hydra-core>=1.3.0     # Hierarchical config management
tqdm>=4.65.0
loguru>=0.7.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Data Storage
h5py>=3.9.0           # HDF5 for transition datasets
```

---

## 9. Configuration System

### `config/environment.yaml`

```yaml
environment:
  n_satellites: 3
  orbital:
    semi_major_axis: 6778.0   # km (400 km altitude)
    eccentricity: 0.0
    inclination: 51.6         # degrees (ISS-like)
    dt: 10.0                  # seconds per step
    period: 92.0              # minutes (one orbit)
  
  battery:
    capacity_wh: 100.0        # Wh
    soc_min: 0.15             # 15% minimum
    soc_max: 1.0              # 100%
    SoH_initial: 1.0
    charge_efficiency: 0.95
    discharge_efficiency: 0.92
  
  thermal:
    T_nominal: 20.0           # Celsius
    T_max: 60.0               # Safety limit
    T_min: -20.0              # Safety limit
  
  solar:
    panel_area: 2.0           # m²
    efficiency: 0.28          # 28% solar cells
    solar_constant: 1361.0    # W/m²
```

### `config/training.yaml`

```yaml
training:
  algorithm: MAPPO
  n_episodes: 10000
  episode_length: 1000
  batch_size: 512
  lr_actor: 3.0e-4
  lr_critic: 1.0e-3
  gamma: 0.99
  lambda_gae: 0.95
  clip_epsilon: 0.2
  entropy_coef: 0.01
  max_grad_norm: 0.5
  n_epochs_per_update: 10
  
  world_model:
    hidden_dim: 256
    n_layers: 3
    predict_horizon: 5
    learning_rate: 1.0e-3
    n_pretrain_epochs: 100
```

---

## 10. Training Pipeline

### Step 1: Pretrain World Model

```bash
python training/train_world_model.py \
  --config config/training.yaml \
  --n_samples 100000 \
  --epochs 200
```

### Step 2: Train MARL Agents

```bash
python training/train_marl.py \
  --config config/training.yaml \
  --world_model_ckpt checkpoints/world_model_best.pt \
  --n_satellites 3 \
  --episodes 10000
```

### Step 3: Curriculum Training

```bash
python training/curriculum_training.py \
  --stages 1,2,3,4,5,6 \
  --auto_advance
```

### Step 4: Evaluate

```bash
python evaluation/experiment_runner.py \
  --agents checkpoints/marl_final.pt \
  --scenarios all \
  --seeds 5
```

### Step 5: Generate Plots

```bash
python visualization/prediction_plots.py --output figures/
```

### Makefile Shortcuts

```makefile
train-all:
    python training/train_world_model.py && \
    python training/train_marl.py && \
    python training/curriculum_training.py

evaluate:
    python evaluation/experiment_runner.py --scenarios all

dashboard:
    python visualization/dashboard.py --port 8050

test:
    pytest tests/ -v --cov=.
```

---

## 11. Evaluation & Benchmarking

### Experiment Suite

| Experiment | File | Description |
|---|---|---|
| Nominal Conditions | `nominal_conditions.py` | Standard orbit, no anomalies |
| Eclipse Stress Test | `eclipse_stress_test.py` | Extended eclipses, high power demand |
| Degradation Test | `degradation_test.py` | Long-run SoH degradation over 1000 orbits |
| Constellation Scaling | `constellation_scaling.py` | 3 → 6 → 12 satellites |

### Ablation Study

| Configuration | Description |
|---|---|
| CASC-RL (Full) | All layers active |
| No World Model | Random-action exploration, no prediction |
| No Cooperative Layer | Independent PPO agents |
| No Hierarchical Layer | MARL only, no coordinator |
| Rule-Based Baseline | Fixed charge/discharge threshold |
| PID Baseline | Classical PID power controller |

---

## 12. Experiments & Paper Outputs

| Figure | Content | Source Script |
|---|---|---|
| Figure 1 | System Architecture Diagram | (static diagram) |
| Figure 2 | Learning Curves (reward vs. episode) | `prediction_plots.py` |
| Figure 3 | Battery SoC/SoH over mission lifetime | `degradation_test.py` |
| Figure 4 | Mission Completion Rate Comparison | `experiment_runner.py` |
| Figure 5 | Constellation Coordination Visualization | `constellation_view.py` |
| Figure 6 | World Model Prediction Accuracy | `prediction_plots.py` |

### Expected Results

| Metric | Rule-Based | PID | Ind. PPO | CASC-RL |
|---|---|---|---|---|
| Battery Lifetime (orbits) | ~800 | ~900 | ~1050 | **~1400** |
| Mission Success Rate | 62% | 67% | 78% | **91%** |
| Thermal Violations | 24 | 18 | 9 | **2** |
| SoC Min (Stress Test) | 8% | 12% | 18% | **24%** |

---

## 13. Safety & Fault Recovery

### Safety State Machine

```
NOMINAL ──(SoC < 30%)──► WARNING ──(SoC < 15%)──► CRITICAL ──► EMERGENCY_HIBERNATE
   ▲                                                                     │
   └──────────────────── (SoC > 50%) ──────────── RECOVERY ◄────────────┘
```

### Safety Constraints

| Constraint | Threshold | Action |
|---|---|---|
| Battery SoC | < 15% | Force hibernate |
| Battery SoH | < 60% | Reduce max discharge rate |
| Temperature | > 60°C | Thermal safe mode |
| Temperature | < -20°C | Heater ON |
| Eclipse predicted | Within 5 min | Pre-charge buffer |

### Anomaly Injection Tests

```python
# Test 1: Extended eclipse (2× normal duration)
env.inject_anomaly("extended_eclipse", duration=180*60)

# Test 2: Solar panel partial failure
env.inject_anomaly("solar_degradation", efficiency_factor=0.4)

# Test 3: Battery cell failure
env.inject_anomaly("battery_failure", cell_loss=0.3)
```

---

## 14. Visualization & Dashboard

### Dashboard Features

- **Live Satellite Status Panel:** SoC, SoH, Temperature, Mode per satellite
- **Orbital Map:** Real-time 3D constellation position with eclipse zones
- **Reward History:** Rolling average reward for each agent
- **World Model Accuracy:** Predicted vs. actual state comparison
- **Mission Control:** Issue override commands, inject anomalies
- **Experiment Mode:** Run and compare baseline configurations

**Running the Dashboard:**

```bash
python visualization/dashboard.py --port 8050
# Open: http://localhost:8050
```

---

## 15. Production Deployment

### Docker Setup

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "training/train_marl.py"]
```

```bash
# Build and run
docker build -t casc-rl .
docker run --gpus all -v ./checkpoints:/app/checkpoints casc-rl

# With docker-compose for dashboard + training
docker-compose up
```

### Ground Station API

```python
# api/ground_station.py (Flask REST API)
@app.route('/constellation/status', methods=['GET'])
def get_status():
    return jsonify(coordinator.get_global_state())

@app.route('/satellite/<int:sat_id>/command', methods=['POST'])
def send_command(sat_id):
    command = request.json['command']
    satellite_agents[sat_id].receive_command(command)
    return {'status': 'sent'}
```

### Model Export (ONNX)

```python
# Export policy network for edge deployment (onboard computer)
torch.onnx.export(
    actor_net,
    (sample_state,),
    "models/policy_sat1.onnx",
    opset_version=14
)
```

---

## 16. Testing Strategy

### Unit Tests

| Test File | Coverage |
|---|---|
| `test_environment.py` | Physics models, Gym compliance |
| `test_world_model.py` | Prediction accuracy, loss convergence |
| `test_agents.py` | Policy output shapes, safety filter |
| `test_marl.py` | MAPPO update correctness, buffer sampling |
| `test_coordination.py` | Task allocation feasibility |

### Integration Tests

```bash
# Run full pipeline on minimal config (quick smoke test)
pytest tests/integration/ -k "smoke" --timeout=60

# Full integration test (slower)
pytest tests/integration/ --timeout=600
```

### Regression Tests

- L1 physics: Compare vs. analytical orbit predictions
- World model: Prediction error < 5% on test dataset
- MARL: Policy reward improves over random baseline

---

## 17. Contribution Guide & Research Extensions

### Extending the System

| Extension | Where to Add |
|---|---|
| New satellite action type | `agents/action_selector.py` |
| New reward component | `marl/cooperative_rewards.py` |
| New physics model | `environment/` (new module + integrate in `constellation_env.py`) |
| New MARL algorithm | `marl/` (new trainer file) |
| New experiment scenario | `experiments/` |

### Planned Future Work

- [ ] **Model-Based RL (Dreamer-V3):** Replace MAPPO with latent-space imagination
- [ ] **Graph Neural Networks (GNN) for Communication:** Replace broadcast with GNN message passing
- [ ] **Real Telemetry Integration:** Connect to COSMOS or OpenMCT ground station
- [ ] **Federated Learning:** On-orbit decentralized model update without ground downlink
- [ ] **Curriculum RL for Adversarial Training:** Train against worst-case eclipse sequences

### Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{casc-rl-2026,
  title   = {CASC-RL: Cognitive Autonomous Satellite Constellation with Reinforcement Learning},
  author  = {[Author Names]},
  year    = {2026},
  note    = {GitHub repository: https://github.com/[your-repo]/casc-rl}
}
```

---

## Appendix A: Key Notation Reference

| Symbol | Meaning |
|---|---|
| `s_t` | Environment state at time t |
| `a_t` | Action taken at time t |
| `ŝ_{t+k}` | World model prediction k steps ahead |
| `fψ` | World model neural network (parameters ψ) |
| `πi` | Policy of satellite i |
| `V` | Centralized value function (critic) |
| `A_t` | Advantage estimate at time t |
| `r_t` | Probability ratio (new/old policy) |
| `SoC` | State of Charge (0–1) |
| `SoH` | State of Health (0–1, capacity retention) |
| `T` | Temperature (°C) |
| `ε` | PPO clipping parameter (default 0.2) |
| `γ` | Discount factor (default 0.99) |
| `λ` | GAE lambda parameter (default 0.95) |

---

## Appendix B: Quickstart

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

---

*Document Version: 1.0 | Last Updated: March 2026 | Project: CASC-RL*
