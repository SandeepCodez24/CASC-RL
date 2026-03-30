
/**
 * CASC-RL Simulation Engine
 * Drives the real-time satellite constellation simulation.
 * Sources: constellation_env.py physics + world_model.py rollout + action_selector.py safety gate.
 *
 * Observation vector (8-dim, matches Python training env):
 *   [0] SoC           (0-1)
 *   [1] SoH           (0-1)
 *   [2] temperature   normalized (raw°C / 100)
 *   [3] solar_input   normalized (0-1)
 *   [4] orbital_phase (0-1)
 *   [5] eclipse_flag  (0 or 1)
 *   [6] power_consumption (0-1)
 *   [7] comm_delay    (0-1)
 */

import * as ort from 'onnxruntime-web';
import { propagateOrbit, isSatelliteInEclipse } from './physics.js';

// ─── Constants matching Python training env ───────────────────────────────────
export const MU_EARTH   = 3.986004418e14;
export const OBS_DIM    = 8;
export const N_ACTIONS  = 5;
export const PREDICT_K  = 5;

export const ACTION_NAMES = ['PAYLOAD_ON','PAYLOAD_OFF','HIBERNATE','RELAY_MODE','CHARGE_PRIORITY'];
export const ACTION_MODES = ['PAYLOAD', 'PAYLOAD_OFF', 'HIBERNATE', 'RELAY', 'CHARGE'];

// Safety constraints (action_selector.py)
const SOC_CRITICAL    = 0.10;
const SOC_MIN         = 0.15;
const TEMP_MAX_NORM   = 0.60;  // 60°C normalized
const TEMP_WARN_NORM  = 0.50;  // 50°C normalized

// World Model score weights (Algorithm 4)
const W1_SOC     = 1.0;
const W2_DEGRAD  = 0.5;
const W3_THERMAL = 0.3;

// State normalizer stats (approximate, matching training distribution)
const STATE_MEAN = [0.65, 0.95, 0.32, 0.45, 0.50, 0.25, 0.35, 0.20];
const STATE_STD  = [0.20, 0.04, 0.10, 0.25, 0.29, 0.43, 0.15, 0.12];

// ─── ONNX Session State ───────────────────────────────────────────────────────
let wmSession    = null;
let actorSession = null;
let modelsReady  = false;

export async function initModels() {
  try {
    wmSession    = await ort.InferenceSession.create('/world_model.onnx', { executionProviders: ['wasm'] });
    actorSession = await ort.InferenceSession.create('/mappo.onnx',      { executionProviders: ['wasm'] });
    modelsReady  = true;
    console.log('[CASC-RL] ONNX Models loaded — WorldModel + MAPPO Actor ONLINE');
    return true;
  } catch(e) {
    console.warn('[CASC-RL] ONNX load failed, using rule-based fallback:', e.message);
    return false;
  }
}

export function areModelsReady() { return modelsReady; }

// ─── State Normalization ──────────────────────────────────────────────────────
function normalizeObs(obs) {
  return obs.map((v, i) => (v - STATE_MEAN[i]) / (STATE_STD[i] + 1e-8));
}

// ─── World Model: k-step rollout ──────────────────────────────────────────────
export async function worldModelRollout(obs_norm, actionIdx, k = PREDICT_K) {
  // Returns array of k predicted normalized states
  const futures = [];
  let s = [...obs_norm];

  if (!wmSession) {
    // Fallback: simple linear extrapolation
    for (let i = 0; i < k; i++) {
      const next = [...s];
      if (actionIdx === 4) next[0] = Math.min(1, s[0] + 0.02 * (i + 1)); // CHARGE
      else if (actionIdx === 0) next[0] = Math.max(0, s[0] - 0.015 * (i + 1)); // PAYLOAD_ON
      else if (actionIdx === 2) next[0] = Math.max(0, s[0] - 0.003 * (i + 1)); // HIBERNATE
      futures.push(next);
    }
    return futures;
  }

  for (let i = 0; i < k; i++) {
    try {
      const sTensor = new ort.Tensor('float32', new Float32Array(s), [1, OBS_DIM]);
      // Try int64 first (matches export), fallback to int32
      let aTensor;
      try {
        aTensor = new ort.Tensor('int64', new BigInt64Array([BigInt(actionIdx)]), [1]);
      } catch(_) {
        aTensor = new ort.Tensor('int32', new Int32Array([actionIdx]), [1]);
      }
      const out = await wmSession.run({ state: sTensor, action: aTensor });
      s = Array.from(out.next_state_mean?.data || out[Object.keys(out)[0]].data);
      futures.push([...s]);
    } catch(e) {
      // fallback: linear extrapolation on error
      const next = [...s];
      if (actionIdx === 4) next[0] = Math.min(1, s[0] + 0.015);
      else if (actionIdx === 0) next[0] = Math.max(0, s[0] - 0.012);
      futures.push(next);
      s = next;
    }
  }
  return futures;
}

// ─── MPC Scoring (Algorithm 4) ────────────────────────────────────────────────
function scoreState(s_t_norm, s_future_norm) {
  // Denormalize just the relevant dims for scoring
  const soc    = s_future_norm[0] * STATE_STD[0] + STATE_MEAN[0];
  const soh_t  = s_t_norm[1]      * STATE_STD[1] + STATE_MEAN[1];
  const soh_f  = s_future_norm[1] * STATE_STD[1] + STATE_MEAN[1];
  const temp   = s_future_norm[2] * STATE_STD[2] + STATE_MEAN[2];
  const soh_loss = Math.max(0, soh_t - soh_f);
  return W1_SOC * soc - W2_DEGRAD * soh_loss - W3_THERMAL * temp;
}

// ─── Actor ONNX inference ─────────────────────────────────────────────────────
async function actorInference(obs_norm, futures_norm) {
  if (!actorSession) return null;
  const s_t_arr = new Float32Array(obs_norm);
  const s_future_flat = new Float32Array(PREDICT_K * OBS_DIM);
  futures_norm.forEach((f, i) => f.forEach((v, j) => { s_future_flat[i * OBS_DIM + j] = v; }));

  const s_t_tensor      = new ort.Tensor('float32', s_t_arr,      [1, OBS_DIM]);
  const s_future_tensor = new ort.Tensor('float32', s_future_flat,[1, PREDICT_K, OBS_DIM]);

  const results = await actorSession.run({ s_t: s_t_tensor, s_future: s_future_tensor });
  const logits  = Array.from(results.logits.data);
  return logits;
}

// ─── Safety Gate (action_selector.py logic) ──────────────────────────────────
function safetyGate(policyAction, sat) {
  const soc  = sat.soc / 100;
  const temp = sat.temp / 100;  // normalized like Python env

  if (soc < SOC_CRITICAL)
    return { action: 4, overridden: true, reason: 'SoC CRITICAL — Forced CHARGE' };
  if (soc < SOC_MIN && policyAction === 0)
    return { action: 1, overridden: true, reason: 'SoC below min — Payload denied' };
  if (temp > TEMP_MAX_NORM)
    return { action: 2, overridden: true, reason: 'Temperature CRITICAL — Hibernate' };
  if (temp > TEMP_WARN_NORM && policyAction === 0)
    return { action: 1, overridden: true, reason: 'Temp WARNING — Payload denied' };

  return { action: policyAction, overridden: false, reason: 'nominal' };
}

// ─── Full Per-Satellite Cognitive Decision ────────────────────────────────────
export async function cognitiveDecision(sat, simTime) {
  // Build obs vector
  const eclipseFlag = isSatelliteInEclipse(propagateOrbit(sat.elements, simTime)) ? 1 : 0;
  const orbPhase    = ((simTime % 5640) / 5640); // 94-min LEO period approx

  const obs_raw = [
    sat.soc / 100,           // [0] SoC
    sat.soh,                  // [1] SoH
    sat.temp / 100,           // [2] Temp normalized
    eclipseFlag ? 0 : 0.8,   // [3] Solar input (0 if eclipse)
    orbPhase,                 // [4] Orbital phase
    eclipseFlag,              // [5] Eclipse flag
    sat.mode === 'PAYLOAD' ? 0.5 : sat.mode === 'HIBERNATE' ? 0.05 : 0.2, // [6] Power
    0.15,                     // [7] Comm delay (fixed for demo)
  ];
  const obs_norm = normalizeObs(obs_raw);

  // MPC scoring across all 5 candidate actions
  const scores   = [];
  const allFutures = [];

  for (let a = 0; a < N_ACTIONS; a++) {
    const futures = await worldModelRollout(obs_norm, a, PREDICT_K);
    allFutures.push(futures);
    const terminal = futures[futures.length - 1];
    scores.push(scoreState(obs_norm, terminal));
  }

  // Also get actor logits from MAPPO
  let logits = null;
  try {
    logits = await actorInference(obs_norm, allFutures[1]); // use PAYLOAD_OFF futures
  } catch(_) {}

  // Best action by MPC
  let mpcBest = scores.indexOf(Math.max(...scores));

  // Best action by actor (if available)
  let actorBest = logits ? logits.indexOf(Math.max(...logits)) : mpcBest;

  // Blend: use actor if models loaded, MPC as fallback
  const rawAction = modelsReady ? actorBest : mpcBest;

  // Safety gate
  const { action: safeAction, overridden, reason } = safetyGate(rawAction, sat);

  // Denormalized future SoC predictions for the chosen action
  const chosenFutures = allFutures[safeAction].map(f => ({
    soc:  Math.max(0, Math.min(100, (f[0] * STATE_STD[0] + STATE_MEAN[0]) * 100)),
    temp: Math.max(0, (f[2] * STATE_STD[2] + STATE_MEAN[2]) * 100),
    soh:  Math.max(0, Math.min(1, f[1] * STATE_STD[1] + STATE_MEAN[1])),
    eclipseRisk: f[5] > 0.4,
  }));

  // Eclipse prediction: look ahead in chosen futures for eclipse entry
  let eclipseCountdown = null;
  const futureEclipseIdx = chosenFutures.findIndex(f => f.eclipseRisk);
  if (!eclipseFlag && futureEclipseIdx >= 0) {
    eclipseCountdown = futureEclipseIdx * 60 + 30; // approx seconds
  }

  return {
    rawAction,
    safeAction,
    overridden,
    overrideReason: reason,
    scores,          // MPC scores per action [5]
    logits,          // Actor logits [5] or null
    obs_raw,
    obs_norm,
    chosenFutures,   // k predicted states
    eclipseFlag: !!eclipseFlag,
    eclipseCountdown, // seconds until eclipse, null if none
    mode: ACTION_MODES[safeAction],
  };
}

// ─── Physics Update (simplified constellation_env.py step) ───────────────────
export function physicStep(sat, action, eclipseFlag, dt = 10) {
  let { soc, soh, temp } = sat;
  const mode = ACTION_MODES[action];

  // Solar power (0 in eclipse)
  const P_solar = eclipseFlag ? 0 : 0.85; // normalized

  // Power consumption per mode
  const P_consume = {
    PAYLOAD: 0.60, PAYLOAD_OFF: 0.30, HIBERNATE: 0.05, RELAY: 0.40, CHARGE: 0.20
  }[mode] || 0.30;

  // SoC update (Coulomb counting proxy)
  const dSoc = (P_solar - P_consume) * 0.002 * dt;
  soc = Math.max(0, Math.min(100, soc + dSoc));

  // Temperature update (thermal model proxy)
  const Q_internal = P_consume * 10;
  const Q_radiate  = (temp - 20) * 0.15;
  const Q_solar    = eclipseFlag ? -5 : 8;
  const dTemp      = (Q_internal + Q_solar - Q_radiate) * 0.01 * dt;
  temp = Math.max(-20, Math.min(80, temp + dTemp));

  // SoH degradation (Wöhler model proxy)
  const thermalFactor = Math.max(0, (temp - 35) / 100);
  const cycleFactor   = (mode === 'PAYLOAD' || mode === 'CHARGE') ? 0.00001 : 0.000002;
  soh = Math.max(0.5, soh - cycleFactor * dt * (1 + thermalFactor * 2));

  // Status from SoC thresholds
  const status = soc < 10 ? 'CRITICAL' : soc < 20 ? 'WARNING' : 'NOMINAL';

  return { ...sat, soc, soh, temp, mode, status, eclipseFlag };
}

// ─── Traditional PID / Rule-Based Controller ─────────────────────────────────
export function traditionalControl(sat, eclipseFlag) {
  // Reactive PID: only react to current state, no prediction
  if (sat.soc < 20) return 4; // CHARGE_PRIORITY
  if (sat.temp > 60) return 2;  // HIBERNATE
  return 0; // PAYLOAD_ON by default
}

// ─── Anomaly Injector ─────────────────────────────────────────────────────────
export function injectAnomaly(sat, type) {
  switch(type) {
    case 'solar_fault':
      return { ...sat, solarEfficiency: 0.4, anomaly: 'SOLAR_FAULT' };
    case 'battery_drain':
      return { ...sat, soc: Math.max(0, sat.soc - 40), anomaly: 'BATTERY_DRAIN' };
    case 'thermal_spike':
      return { ...sat, temp: Math.min(80, sat.temp + 30), anomaly: 'THERMAL_SPIKE' };
    case 'clear':
      return { ...sat, solarEfficiency: 1.0, anomaly: null };
    default:
      return sat;
  }
}
