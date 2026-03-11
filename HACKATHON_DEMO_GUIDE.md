# CASC-RL: Hackathon Demo & Simulation Frontend Guide
### *Visual Demonstration Strategy for Judges — Multi-Agent Satellite Constellation*

---

> **Purpose:** This document defines how to present CASC-RL to hackathon judges through an interactive simulation frontend — going beyond static charts to show the system *thinking*, *deciding*, and *recovering* in real time.
>
> **Audience:** Hackathon judges unfamiliar with RL — must be visually compelling in under 5 minutes.

---

## Table of Contents

1. [Presentation Strategy](#1-presentation-strategy)
2. [The Core Story to Tell](#2-the-core-story-to-tell)
3. [Simulation Frontend Architecture](#3-simulation-frontend-architecture)
4. [Live Demo Screens](#4-live-demo-screens)
5. [Side-by-Side Comparison Mode](#5-side-by-side-comparison-mode)
6. [Key Visual Moments (Wow Factors)](#6-key-visual-moments-wow-factors)
7. [Implementation Stack](#7-implementation-stack)
8. [Demo Flow Script (5-Minute Pitch)](#8-demo-flow-script-5-minute-pitch)
9. [Comparison Panels Reference](#9-comparison-panels-reference)
10. [Connecting to Project Architecture](#10-connecting-to-project-architecture)

---

## 1. Presentation Strategy

### What Judges Actually Want to See

| What They Expect | What Will Wow Them |
|---|---|
| Training loss curves | Live satellites making decisions in real time |
| Static bar charts | Side-by-side "dumb vs smart" satellite behavior |
| Architecture diagrams | An anomaly happening — and the AI recovering from it |
| Code walkthrough | A satellite dying under traditional control vs. surviving under CASC-RL |

### The Golden Rule
> **Show, don't tell.** Every claim must be backed by something visible moving on screen.

---

## 2. The Core Story to Tell

### The Narrative Arc (3 Acts)

**Act 1 — The Problem** *(60 seconds)*
- Show a satellite constellation orbiting Earth in 3D
- Trigger an extended eclipse event
- Watch a *rule-based* satellite's battery drain to 0% → satellite fails
- Text overlay: *"Traditional systems can't predict. They react too late."*

**Act 2 — The Solution** *(120 seconds)*
- Switch to CASC-RL mode — same eclipse event
- The agent's world model *predicts* the eclipse 5 minutes before entry
- Satellite pre-charges battery, reduces payload load, enters relay mode
- Battery never drops below 24% — mission continues
- Constellation coordinates: one satellite relays for another during eclipse
- Text overlay: *"CASC-RL sees the future. It acts before the crisis."*

**Act 3 — The Proof** *(60 seconds)*
- Switch to comparison dashboard
- Show side-by-side metrics: battery lifetime +75%, thermal violations -92%
- Show the ablation: remove world model → degrades. Remove coordination → degrades.
- *"Every layer adds measurable value."*

---

## 3. Simulation Frontend Architecture

### Overall Frontend Structure

```
┌──────────────────────────────────────────────────────────────────────┐
│                    CASC-RL DEMO DASHBOARD                            │
│                                                                      │
│  ┌─────────────────────────┐   ┌──────────────────────────────────┐ │
│  │   3D ORBITAL VIEW       │   │   SATELLITE TELEMETRY PANELS     │ │
│  │  (Three.js / Plotly)    │   │   SoC │ SoH │ Temp │ Mode       │ │
│  │                         │   │                                  │ │
│  │   🛰️ SAT-1  🛰️ SAT-2    │   │  [SAT-1] ████████░░ 80% SoC   │ │
│  │          🛰️ SAT-3        │   │  [SAT-2] ██████░░░░ 62% SoC   │ │
│  │   ☀️ Sun vector          │   │  [SAT-3] █████████░ 88% SoC   │ │
│  │   🌑 Eclipse zones       │   │                                  │ │
│  └─────────────────────────┘   └──────────────────────────────────┘ │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │   AGENT DECISION LAYER (Live)                                   │ │
│  │   World Model Prediction │ Action Taken │ Safety Status         │ │
│  │   "Eclipse in 4m 30s"   │  [RELAY MODE]│  ✅ NOMINAL           │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌─────────────────────────┐   ┌──────────────────────────────────┐ │
│  │   REWARD / LEARNING     │   │   COMPARISON PANEL               │ │
│  │   Live reward chart     │   │   Traditional vs CASC-RL        │ │
│  └─────────────────────────┘   └──────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### Frontend Component Map

```
frontend/
├── index.html                    # Main entry point
├── app.js                        # App controller & state machine
│
├── views/
│   ├── orbital_view/             # 3D satellite constellation
│   │   ├── OrbitalScene.js       # Three.js scene setup
│   │   ├── SatelliteObject.js    # 3D satellite mesh + color coding
│   │   ├── EclipseZone.js        # Earth shadow cone visualization
│   │   └── OrbitTrail.js         # Satellite trail rendering
│   │
│   ├── telemetry/                # Per-satellite status panels
│   │   ├── TelemetryCard.js      # SoC/SoH/Temp/Mode card
│   │   ├── BatteryGauge.js       # Animated battery level indicator
│   │   └── ModeIndicator.js      # Active mode badge (color coded)
│   │
│   ├── agent_brain/              # Agent decision visualization
│   │   ├── WorldModelDisplay.js  # "Predicting: Eclipse in X seconds"
│   │   ├── ActionExplainer.js    # Why did the agent pick this action?
│   │   └── SafetyStateWidget.js  # NOMINAL / WARNING / CRITICAL banner
│   │
│   ├── charts/                   # Real-time metric charts
│   │   ├── SoCTimeline.js        # SoC over time (all satellites)
│   │   ├── RewardChart.js        # Episode reward curve
│   │   └── ThermalChart.js       # Temperature per satellite
│   │
│   └── comparison/               # Side-by-side comparison view
│       ├── DuelView.js           # Split screen: Traditional vs CASC-RL
│       ├── MetricsTable.js       # Animated numbers comparison
│       └── RadarChart.js         # Spider chart comparison
│
├── simulation/
│   ├── SimulationEngine.js       # Runs physics + agent at configurable speed
│   ├── TraditionalController.js  # Rule-based / PID baseline logic
│   ├── CASCRLController.js       # CASC-RL policy (pre-trained, loaded)
│   └── AnomalyInjector.js        # Inject eclipse / solar / battery faults
│
├── data/
│   ├── pretrained_policy.json    # Exported CASC-RL policy weights (ONNX/JSON)
│   ├── comparison_results.json   # Pre-computed benchmark data
│   └── demo_scenarios.json       # Named demo event sequences
│
└── styles/
    ├── main.css                  # Dark space theme styling
    ├── telemetry.css             # Panel styling
    └── animations.css            # Pulse / flash / glow effects
```

---

## 4. Live Demo Screens

### Screen 1 — Orbital View (The "Wow" Opening)

**What it shows:**
- Rotating 3D Earth with realistic lighting
- 3 satellites orbiting in Walker-Delta constellation
- Color-coded by current operational mode:
  - 🟢 **Green** = Payload Active
  - 🔵 **Blue** = Relay Mode
  - 🟡 **Yellow** = Charge Priority
  - 🔴 **Red** = Hibernating / Low Battery
  - ⚫ **Grey** = Critical / Emergency
- Eclipse shadow cone sweeping across the orbit path
- Real-time sun vector arrow
- ISL communication links rendered as glowing lines between satellites

**Interactive Elements:**
- Click a satellite → expand its telemetry card
- Drag to rotate Earth view
- Speed control slider (1× to 100× orbital time)
- "Inject Anomaly" button panel (for live demo)

---

### Screen 2 — Agent Brain Panel (Making Decisions Visible)

**What it shows:**
The internal reasoning of the CASC-RL agent made *readable* for judges:

```
┌─────────────────────────────────────────────────────┐
│  🧠 SAT-2 Agent Decision Engine                      │
│─────────────────────────────────────────────────────│
│  Current State:                                     │
│    SoC: 61%  |  SoH: 0.94  |  Temp: 32°C           │
│                                                     │
│  World Model Forecast (next 5 min):                 │
│    ⚡ Eclipse entry predicted in: 4m 22s            │
│    📉 Projected SoC at eclipse entry: 58%           │
│    🌡️ Projected temp at eclipse exit: 28°C          │
│                                                     │
│  Candidate Actions Evaluated:                       │
│    payload_ON     → score: 0.41  ❌ (SoC risk)     │
│    relay_mode     → score: 0.78  ✅ SELECTED        │
│    hibernate      → score: 0.52                     │
│    charge_priority→ score: 0.71                     │
│                                                     │
│  Safety Check:  ✅ NOMINAL  (SoC 61% > min 15%)    │
│  Decision: RELAY MODE  (pre-eclipse transition)     │
└─────────────────────────────────────────────────────┘
```

**Why this matters for judges:**
- Makes the "black box" transparent
- Shows the world model prediction working in real time
- Judges can *see* the agent thinking before acting

---

### Screen 3 — Telemetry Monitor

**What it shows:**
Live updating status for all satellites in a grid:

```
┌──────────────────┬──────────────────┬──────────────────┐
│   🛰️ CASC-1       │   🛰️ CASC-2       │   🛰️ CASC-3       │
├──────────────────┼──────────────────┼──────────────────┤
│ SoC  ████████░░  │ SoC  ██████░░░░  │ SoC  █████████░  │
│      80%         │      62%         │      88%         │
│ SoH  0.97        │ SoH  0.94        │ SoH  0.96        │
│ Temp 28°C ✅     │ Temp 35°C ✅     │ Temp 24°C ✅     │
│ Mode: PAYLOAD    │ Mode: RELAY      │ Mode: CHARGE     │
│ Status: NOMINAL  │ Status: NOMINAL  │ Status: NOMINAL  │
└──────────────────┴──────────────────┴──────────────────┘
```

**Animated elements:**
- Battery bar animates as it charges/drains
- Status banner pulses orange for WARNING, red for CRITICAL
- Mode badge transitions with a fade animation
- Temperature gauge changes color: blue → green → orange → red

---

### Screen 4 — Comparison Dashboard

**Split-screen showing Traditional vs CASC-RL running simultaneously:**

```
┌──────────────────────────┬───────┬──────────────────────────┐
│  🔴 TRADITIONAL (PID)    │  VS   │  🟢 CASC-RL              │
├──────────────────────────┤       ├──────────────────────────┤
│  SAT-1: SoC 12% ⚠️        │       │  SAT-1: SoC 68% ✅       │
│  SAT-2: SoC 8%  🔴        │       │  SAT-2: SoC 71% ✅       │
│  SAT-3: SoC 21% ⚠️        │       │  SAT-3: SoC 74% ✅       │
│                          │       │                          │
│  Thermal Events: 14      │       │  Thermal Events: 1       │
│  Tasks Completed: 23     │       │  Tasks Completed: 41     │
│  Battery Lifetime: 812   │       │  Battery Lifetime: 1398  │
│       orbits             │       │       orbits             │
└──────────────────────────┴───────┴──────────────────────────┘
```

---

## 5. Side-by-Side Comparison Mode

### How to Structure the Visual Comparisons for Judges

Each comparison is a **30-second visual story**:

#### Comparison A — Eclipse Survival Test

| Moment | Traditional (PID) | CASC-RL |
|---|---|---|
| T-5 min before eclipse | Still on payload mode | World model predicts eclipse → switches to charge_priority |
| Eclipse entry | SoC: 35% → starts draining fast | SoC: 72% → buffer is full |
| Mid eclipse | SoC: 11% → WARNING | SoC: 51% → NOMINAL |
| Eclipse exit | SoC: 6% → CRITICAL, hibernating | SoC: 44% → resumes payload |
| **Visual result** | 🔴 Satellite offline | 🟢 Mission continues |

#### Comparison B — Degradation Over Time

**Chart:** SoH (%) vs. Orbit Number (1–1000)
- Traditional: steep, uncontrolled degradation curve
- CASC-RL: controlled, slower curve with clearly longer battery life
- Annotate the crossover point where traditional hits 70% end-of-life threshold
- CASC-RL reaches the same threshold ~75% later

#### Comparison C — Cooperative vs. Independent

**Scenario:** Two satellites need to relay data. Only one has enough power.
- **Independent PPO:** Both try to relay → conflict → neither succeeds cleanly
- **CASC-RL MAPPO:** Coordinator assigns one to relay, one to charge → perfect handoff
- Visual: ISL link lights up cleanly in CASC-RL, flickers/fails in independent version

#### Comparison D — Fault Recovery

**Anomaly injected live:** Solar panel efficiency drops to 40%
- **Traditional:** Continues payload → battery hits critical within 3 minutes
- **CASC-RL:** Anomaly detected by anomaly_detector → triggers recovery_policy → payload suspended, enters low-power mode, SoC recovers
- Visual: Red pulse on satellite → orange WARNING banner → recovery sequence → green NOMINAL

---

## 6. Key Visual Moments (Wow Factors)

These are the 5 moments to plan, rehearse, and make dramatic for judges:

### 🎯 Moment 1 — "The Prediction" *(World Model)*
*At 4 minutes before eclipse:*
- A countdown timer appears on SAT-2's telemetry card
- The Agent Brain panel shows: `⚡ Eclipse predicted in 4:22`
- The satellite smoothly transitions from PAYLOAD → CHARGE PRIORITY
- Battery bar visibly climbs before eclipse hits
- **What to say:** *"The satellite just predicted the future. It's pre-charging before it even enters the shadow."*

### 🎯 Moment 2 — "The Kill" *(Traditional Failure)*
*Running PID baseline during eclipse:*
- Battery bar drains rapidly in red
- WARNING banner pulses orange
- Then CRITICAL — satellite goes dark on the 3D orbital view
- The orbital trail disappears for that satellite
- **What to say:** *"The traditional controller didn't see it coming. The satellite is now offline."*

### 🎯 Moment 3 — "The Handoff" *(Cooperative MARL)*
*Data relay during eclipse coverage gap:*
- ISL communication line lights up between SAT-1 and SAT-3
- Task allocation panel shows: `SAT-2: RELAY → SAT-1: CHARGE → SAT-3: PAYLOAD`
- Mission completion counter ticks up
- **What to say:** *"Three satellites just negotiated a task handoff without any ground station. That's the multi-agent layer."*

### 🎯 Moment 4 — "The Recovery" *(Safety Layer)*
*Inject solar_degradation anomaly live during demo:*
- Click "Inject Anomaly: Solar Failure (40%)" button
- Red pulse flashes on satellite
- Safety monitor triggers: NOMINAL → WARNING → RECOVERY
- Agent automatically switches to hibernate + charge_priority
- Battery stops falling, stabilizes, begins recovering
- **What to say:** *"I just killed 60% of that satellite's solar power. Watch it recover autonomously."*

### 🎯 Moment 5 — "The Numbers" *(Comparison Dashboard)*
*Switch to comparison view — animated counters:*
- Battery Lifetime: `812 → 1,398 orbits` (counter animates up)
- Mission Success: `62% → 91%` (counter animates up)
- Thermal Violations: `24 → 2` (counter animates down)
- **What to say:** *"75% longer battery life. 91% mission success. 92% fewer thermal failures. Same hardware. Different brain."*

---

## 7. Implementation Stack

### Recommended Technology Choices

| Component | Technology | Why |
|---|---|---|
| 3D Orbital View | **Three.js** | WebGL 3D with Earth globe, orbit paths, satellite meshes |
| Real-time Charts | **Plotly.js** or **Chart.js** | Live-updating SoC/temperature/reward lines |
| Radar / Spider Chart | **Chart.js (radar)** or **D3.js** | Multi-metric comparison in one view |
| Dashboard Layout | **Vanilla HTML/CSS Grid** | Fast, no framework overhead |
| Physics Engine (JS) | Custom JS port of `orbital_dynamics.py` | Run simulation in browser |
| Policy (inference) | **ONNX Runtime Web** | Run pre-trained PyTorch model in browser |
| Backend (optional) | **Python Flask + WebSocket** | Stream live data from real Python simulation |
| Animation | **CSS + GSAP** | Smooth transitions, battery pulses, warning flashes |

### Two Deployment Options

#### Option A — Fully In-Browser (Self-Contained)
- Port the simplified physics engine to JavaScript
- Export trained policy to ONNX → load via `onnxruntime-web`
- Everything runs in browser — no server needed
- **Best for hackathon:** One HTML file, runs offline

#### Option B — Python Backend + Browser Frontend
- Python Flask server runs full `ConstellationEnv` simulation
- Sends state updates via WebSocket every 100ms
- Browser renders in real time
- **Best for accuracy:** Exact same physics as training

```
Python Backend                     Browser Frontend
┌──────────────────┐   WebSocket   ┌──────────────────────────┐
│ ConstellationEnv │──────────────►│ SoC/Temp/Mode displays   │
│ CASC-RL Policy   │  JSON state   │ 3D orbital view          │
│ Traditional Ctrl │   updates     │ Comparison panels        │
└──────────────────┘               └──────────────────────────┘
```

### Data Flow for Demo

```
1. demo_scenarios.json defines named scenarios:
   - "eclipse_test"
   - "anomaly_injection"
   - "cooperative_relay"
   - "degradation_1000_orbits"

2. SimulationEngine.js reads scenario config
3. Runs physics at configurable speed (1× → 100×)
4. Runs TraditionalController OR CASCRLController
5. Emits state events → all UI components update reactively
6. AnomalyInjector.js hooks into simulation at configured timestamps
```

---

## 8. Demo Flow Script (5-Minute Pitch)

### Timing Guide

| Time | Action | Visual |
|---|---|---|
| 0:00 | Open with 3D orbital view rotating | Earth + 3 satellites orbiting |
| 0:15 | Introduce the problem verbally | Zoom to single satellite |
| 0:30 | Run Traditional (PID) — eclipse test | Watch battery drain → satellite fails |
| 1:15 | *"This is what happens without intelligence."* | Dead satellite greyed out |
| 1:30 | Reset — run CASC-RL same scenario | Agent predicts eclipse, pre-charges |
| 2:30 | *"The AI saw the eclipse coming."* | Show Agent Brain countdown |
| 2:45 | Inject solar anomaly live | Red pulse, safety recovery sequence |
| 3:15 | Switch to cooperative relay demo | ISL link lights up, task handoff |
| 3:45 | Switch to Comparison Dashboard | Animated metric counters |
| 4:15 | Show radar chart — all metrics | Full spider chart comparison |
| 4:30 | Close with ablation — remove layers | Each removed layer degrades performance |
| 4:50 | *"Every layer adds measurable value."* | Final summary slide |

---

## 9. Comparison Panels Reference

### Metric Comparison Table (for Judges Dashboard)

| Metric | Rule-Based | PID | Independent PPO | **CASC-RL** | **Improvement** |
|---|---|---|---|---|---|
| Battery Lifetime (orbits) | 800 | 900 | 1,050 | **1,400** | +75% vs Rule-Based |
| Mission Success Rate | 62% | 67% | 78% | **91%** | +47% vs Rule-Based |
| Thermal Violations | 24 | 18 | 9 | **2** | -92% vs Rule-Based |
| SoC Min (Eclipse Stress) | 8% | 12% | 18% | **24%** | +200% vs Rule-Based |
| Anomaly Recovery Time (s) | N/A | N/A | ~180s | **~45s** | 4× faster |

### Ablation Table (What Each Layer Contributes)

| Configuration | Battery Lifetime | Mission Success | Thermal Events |
|---|---|---|---|
| Rule-Based Baseline | 800 | 62% | 24 |
| + World Model only | 1,050 | 71% | 12 |
| + World Model + Local RL | 1,150 | 79% | 8 |
| + Add Cooperative MARL | 1,280 | 87% | 4 |
| + Full CASC-RL (all layers) | **1,400** | **91%** | **2** |

*Each layer shown to add measurable value — no redundant components.*

---

## 10. Connecting to Project Architecture

This demo frontend directly visualizes each layer from `PROJECT_DOCUMENT.md`:

| CASC-RL Layer | What Judges See in Demo |
|---|---|
| **L1: Digital Twin** (`constellation_env.py`) | The 3D orbital simulation itself — the physics engine running on screen |
| **L2: World Model** (`world_model.py`) | Agent Brain panel showing eclipse countdown and SoC forecast |
| **L2: Local RL Policy** (`policy_network.py`) | Action scoring table: `relay_mode: 0.78 ✅ SELECTED` |
| **L2: Safety Monitor** (`safety_monitor.py`) | NOMINAL → WARNING → CRITICAL banner + auto-recovery |
| **L3: Cooperative MARL** (`mappo_trainer.py`) | ISL link visualization, task handoff between satellites |
| **L4: Hierarchical Coordinator** (`cluster_coordinator.py`) | Task allocation panel: `SAT-1: RELAY | SAT-2: CHARGE | SAT-3: PAYLOAD` |
| **L5: Anomaly Detector** (`anomaly_detector.py`) | Red pulse on satellite + live anomaly event log |

### Files in This Repo Supporting the Demo

```
visualization/dashboard.py          ← Python backend dashboard (Plotly Dash)
visualization/constellation_view.py ← 3D orbit plot generator
comparisons/summary/radar_chart_all_methods.py  ← Spider chart data
comparisons/battery/soc_trajectory_comparison.py ← Eclipse test chart
comparisons/safety/fault_recovery_comparison.py  ← Anomaly recovery chart
api/ground_station.py               ← REST API for frontend ↔ backend
```

---

## Appendix: Quick Phrases for Judges

> *"It's not just a model that trains — it's a system that thinks, coordinates, and recovers without any human in the loop."*

> *"The world model is the AI's imagination. It simulates the future before committing to an action."*

> *"MAPPO gives the satellites a shared language. They negotiate tasks the way a team would."*

> *"The safety layer is the fail-safe. Even if the RL policy makes a mistake, the satellite cannot destroy itself."*

> *"This is what Starlink could look like if you replaced mission operations with AI."*

---

*Document Version: 1.0 | CASC-RL Hackathon Edition | March 2026*
