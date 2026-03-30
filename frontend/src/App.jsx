
/**
 * CASC-RL Mission Control Dashboard
 * ─────────────────────────────────
 * Root application: drives real-time satellite simulation using trained
 * ONNX world model + MAPPO actor. Architecture:
 *
 *   12-sat Walker-Delta constellation
 *   → per-step cognitiveDecision() via ONNX inference (world model + actor)
 *   → physicStep() updates SoC/SoH/Temp
 *   → React state → CelestialView + AgentBrain + Telemetry + Comparison + DuelView
 */

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import './styles/design-system.css';

import {
  Orbit, Cpu, LayoutGrid, BarChart3, Scale, Satellite, Activity,
  ShieldCheck, ShieldAlert, AlertTriangle, Play, Pause, RotateCw,
  FastForward, Zap, Radio, Database, ChevronRight, Menu, X
} from 'lucide-react';

import CelestialView    from './components/CelestialView';
import TelemetryPanel   from './components/Telemetry';
import AgentBrain       from './components/AgentBrain';
import ComparisonPanel  from './components/Comparison';
import DuelView         from './components/Comparison/DuelView';

import {
  initModels, areModelsReady, cognitiveDecision,
  physicStep, injectAnomaly,
  ACTION_MODES,
} from './engine/simulation.js';
import { walkerDelta, isSatelliteInEclipse, propagateOrbit, timeToEclipse } from './engine/physics.js';

// ─── Satellite colors ─────────────────────────────────────────────────────────
const SAT_COLORS = [
  '#00f5ff','#7b61ff','#00ff88','#ffaa00','#ff3d3d','#a78bfa',
  '#34d399','#f59e0b','#60a5fa','#f472b6','#4ade80','#fb923c',
];

// ─── Initial constellation ────────────────────────────────────────────────────
function buildConstellation() {
  const elements = walkerDelta(12, 7000e3, 0.9);
  return elements.map((el, i) => ({
    id:       `CASC-${i + 1}`,
    color:    SAT_COLORS[i],
    elements: el,
    soc:      65 + Math.random() * 20,
    soh:      0.94 + Math.random() * 0.06,
    temp:     24 + Math.random() * 12,
    mode:     'PAYLOAD',
    status:   'NOMINAL',
    eclipseFlag: false,
    anomaly:  null,
    solarEfficiency: 1.0,
    overriddenThisStep: false,
  }));
}

const INITIAL_SATS = buildConstellation();

// ─── Nav items ────────────────────────────────────────────────────────────────
const NAV = [
  { id:'ORBITAL',     label:'Orbital View',    icon:Orbit     },
  { id:'BRAIN',       label:'Agent Brain',     icon:Cpu       },
  { id:'TELEMETRY',   label:'Telemetry Grid',  icon:LayoutGrid},
  { id:'COMPARISON',  label:'Metrics',         icon:BarChart3 },
  { id:'DUEL',        label:'Side-by-Side',    icon:Scale     },
];

export default function App() {
  const [screen,        setScreen]        = useState('ORBITAL');
  const [sats,          setSats]          = useState(INITIAL_SATS);
  const [simTime,       setSimTime]       = useState(0);
  const [simSpeed,      setSimSpeed]      = useState(1);
  const [paused,        setPaused]        = useState(false);
  const [selectedSat,   setSelectedSat]   = useState('CASC-1');
  const [modelsStatus,  setModelsStatus]  = useState('loading'); // loading|ready|fallback
  const [decisions,     setDecisions]     = useState({});        // {satId: decision}
  const [socHistories,  setSocHistories]  = useState({});        // {satId: number[]}
  const [eventLog,      setEventLog]      = useState([
    { t:0, msg:'Initializing CASC-RL ONNX runtime…', level:'info' }
  ]);
  const simTimeRef = useRef(0);

  // Keep ref in sync with state
  useEffect(() => { simTimeRef.current = simTime; }, [simTime]);

  // ─── Load ONNX models ───────────────────────────────────────────────────────
  useEffect(() => {
    initModels().then(ok => {
      if (ok) {
        setModelsStatus('ready');
        addLog(0, 'MAPPO Actor + World Model LOADED via ONNX Runtime ✓', 'success');
      } else {
        setModelsStatus('fallback');
        addLog(0, 'Models unavailable — MPC rule-based fallback active', 'warn');
      }
    });
  }, []);

  const addLog = useCallback((t, msg, level = 'info') => {
    setEventLog(prev => [{ t, msg, level }, ...prev].slice(0, 60));
  }, []);

  // ─── SoC History ring buffer ────────────────────────────────────────────────
  const updateSocHistory = useCallback((id, soc) => {
    setSocHistories(prev => {
      const hist = prev[id] || [];
      return { ...prev, [id]: [...hist.slice(-49), soc] };
    });
  }, []);

  // ─── Main simulation loop ────────────────────────────────────────────────────
  useEffect(() => {
    if (paused) return;
    const TICK_MS = Math.max(80, 150 / simSpeed);

    const interval = setInterval(() => {
      const dt = 10 * simSpeed;
      const currentT = simTimeRef.current;
      setSimTime(t => {
        simTimeRef.current = t + dt;
        return t + dt;
      });

      setSats(prevSats => {
        return prevSats.map(sat => {
          const pos     = propagateOrbit(sat.elements, currentT);
          const eclipse = isSatelliteInEclipse(pos);
          const updated = physicStep(sat, sat._actionIdx ?? 1, eclipse, dt);
          updateSocHistory(sat.id, updated.soc);
          return { ...updated, eclipseFlag: eclipse };
        });
      });
    }, TICK_MS);

    return () => clearInterval(interval);
  }, [paused, simSpeed, updateSocHistory]);

  // ─── Inference loop (runs one satellite per tick, round-robin) ──────────────
  const inferenceRef  = useRef(false);
  const inferTickRef  = useRef(0);

  useEffect(() => {
    if (paused) return;

    const inferenceInterval = setInterval(async () => {
      if (inferenceRef.current) return;  // skip if previous inference still running
      inferenceRef.current = true;

      try {
        const currentSats = sats;
        const satIdx = inferTickRef.current % currentSats.length;
        const sat = currentSats[satIdx];
        inferTickRef.current += 1;

        const decision = await cognitiveDecision(sat, simTimeRef.current);

        // Store decision for display
        setDecisions(prev => ({ ...prev, [sat.id]: decision }));

        // Apply the mode to this satellite
        setSats(ss => ss.map(s => {
          if (s.id !== sat.id) return s;
          if (decision.overridden && !s.overriddenThisStep) {
            addLog(simTimeRef.current, `${sat.id}: Safety override → ${ACTION_MODES[decision.safeAction]} (${decision.overrideReason})`, 'warn');
          }
          if (!s.eclipseFlag && decision.eclipseFlag) {
            addLog(simTimeRef.current, `${sat.id}: 🌑 Eclipse entered`, 'info');
          }
          if (decision.eclipseCountdown !== null && decision.eclipseCountdown < 120 && !s._eclipseWarned) {
            addLog(simTimeRef.current, `${sat.id}: ⚡ Eclipse in ${Math.floor(decision.eclipseCountdown)}s — Pre-charging`, 'warn');
          }
          return {
            ...s,
            mode: ACTION_MODES[decision.safeAction],
            _actionIdx: decision.safeAction,
            overriddenThisStep: decision.overridden,
            _eclipseWarned: decision.eclipseCountdown !== null && decision.eclipseCountdown < 120,
          };
        }));
      } catch (err) {
        console.warn('[Inference] Error:', err.message);
      } finally {
        inferenceRef.current = false;
      }
    }, 600);  // run inference every 600ms

    return () => clearInterval(inferenceInterval);
  }, [paused, sats, addLog]);

  // ─── ISL links ───────────────────────────────────────────────────────────────
  const islLinks = useMemo(() => {
    const links = [];
    for (let i = 0; i < sats.length; i++) {
      const j  = (i + 1) % sats.length;
      const p1 = propagateOrbit(sats[i].elements, simTime);
      const p2 = propagateOrbit(sats[j].elements, simTime);
      const d  = Math.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2+(p1.z-p2.z)**2);
      links.push({
        from:   sats[i].id,
        to:     sats[j].id,
        active: d < 4000e3 && sats[i].mode !== 'HIBERNATE' && sats[j].mode !== 'HIBERNATE',
        relay:  sats[i].mode === 'RELAY' || sats[j].mode === 'RELAY',
      });
    }
    return links;
  }, [sats, simTime]);

  const selectedSatObj = sats.find(s => s.id === selectedSat);
  const selectedDecision = decisions[selectedSat];

  // ─── Aggregate fleet health ───────────────────────────────────────────────
  const fleetHealth = useMemo(() => ({
    nominal:  sats.filter(s => s.status === 'NOMINAL').length,
    warning:  sats.filter(s => s.status === 'WARNING').length,
    critical: sats.filter(s => s.status === 'CRITICAL').length,
    overrides:sats.filter(s => s.overriddenThisStep).length,
    inEclipse:sats.filter(s => s.eclipseFlag).length,
  }), [sats]);

  // ─── Anomaly injection ────────────────────────────────────────────────────
  const handleInjectAnomaly = (type) => {
    setSats(prev => prev.map(s => s.id === selectedSat ? injectAnomaly(s, type) : s));
    addLog(simTime, `ANOMALY INJECT [${selectedSat}]: ${type}`, 'critical');
  };

  const handleReset = () => {
    setSats(buildConstellation());
    setSimTime(0);
    setDecisions({});
    setSocHistories({});
    setEventLog([{ t:0, msg:'Simulation reset', level:'info' }]);
    setInferenceTick(0);
  };

  // ─── Layer state (from decisions) ─────────────────────────────────────────
  const activeModel = modelsStatus === 'ready' ? 'MAPPO/WorldModel ONNX' : 'MPC Fallback';

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <div className="dashboard-root">

      {/* ── HEADER ─────────────────────────────────────────────────────────── */}
      <header className="layout-header glass glass-scan" style={{ display:'flex', alignItems:'center', gap:'16px', padding:'0 16px' }}>
        {/* Brand */}
        <div style={{ display:'flex', alignItems:'center', gap:'10px', flexShrink:0 }}>
          <Satellite size={26} color="var(--holo-primary)" style={{ filter:'drop-shadow(0 0 8px var(--holo-primary))' }} />
          <div>
            <div style={{ fontSize:'1.05rem', fontWeight:700, color:'var(--holo-primary)', fontFamily:"'JetBrains Mono',monospace", lineHeight:1.1 }}>
              CASC-RL
            </div>
            <div style={{ fontSize:'0.52rem', color:'var(--text-dim)', letterSpacing:'2px', textTransform:'uppercase' }}>
              Cognitive Satellite Constellation
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav style={{ display:'flex', gap:'4px', flex:1, justifyContent:'center' }}>
          {NAV.map(n => (
            <button key={n.id} className={`nav-tab ${screen === n.id ? 'active' : ''}`}
              onClick={() => setScreen(n.id)}>
              <n.icon size={14} /> {n.label}
            </button>
          ))}
        </nav>

        {/* Status row */}
        <div style={{ display:'flex', alignItems:'center', gap:'10px', flexShrink:0, fontSize:'0.68rem', fontFamily:"'JetBrains Mono',monospace" }}>
          {/* Model status */}
          <div className={`badge badge-${modelsStatus === 'ready' ? 'holo' : 'warning'}`}>
            <div className="pulse-dot" style={{ background:modelsStatus==='ready'?'var(--holo-primary)':'var(--orange-warn)', width:6, height:6 }} />
            {modelsStatus === 'loading' ? 'LOADING…' : modelsStatus === 'ready' ? 'ONNX ONLINE' : 'FALLBACK'}
          </div>

          {/* Fleet status */}
          <div style={{ display:'flex', gap:'6px' }}>
            <span style={{ color:'var(--green-ok)' }}>{fleetHealth.nominal}✓</span>
            {fleetHealth.warning > 0  && <span style={{ color:'var(--orange-warn)' }}>{fleetHealth.warning}⚠</span>}
            {fleetHealth.critical > 0 && <span style={{ color:'var(--red-crit)', animation:'blink 1s infinite' }}>{fleetHealth.critical}🔴</span>}
          </div>

          <span style={{ color:'var(--text-dim)' }}>
            T: <span className="holo-text">{(simTime/3600).toFixed(2)}h</span>
          </span>

          {/* Speed control */}
          <button className="btn btn-ghost" style={{ fontSize:'0.65rem', padding:'4px 10px' }}
            onClick={() => setSimSpeed(s => s===1?5:s===5?20:1)}>
            {simSpeed}×
          </button>

          {/* Pause/Play */}
          <button className="btn btn-holo" style={{ padding:'4px 10px' }}
            onClick={() => setPaused(p => !p)}>
            {paused ? <Play size={13} /> : <Pause size={13} />}
          </button>

          {/* Reset */}
          <button className="btn btn-ghost" style={{ padding:'4px 10px' }} onClick={handleReset}>
            <RotateCw size={13} />
          </button>
        </div>
      </header>

      {/* ── LEFT SIDEBAR ───────────────────────────────────────────────────── */}
      <aside className="layout-left">
        {/* Satellite fleet */}
        <div className="glass glass-scan" style={{ flex:1, padding:'14px', overflow:'hidden', display:'flex', flexDirection:'column' }}>
          <div className="widget-title">
            <Activity size={13} /> Sat Fleet ({sats.length})
          </div>
          <div style={{ flex:1, overflow:'hidden' }}>
            <TelemetryPanel
              sats={sats}
              selectedId={selectedSat}
              onSelect={setSelectedSat}
              socHistories={socHistories}
            />
          </div>
        </div>

        {/* Coordination layer status */}
        <div className="glass" style={{ padding:'14px', flexShrink:0 }}>
          <div className="widget-title"><Radio size={13} /> Swarm Coordination</div>
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'6px', fontSize:'0.68rem' }}>
            {[
              { label:'ISL Active',  value:`${islLinks.filter(l=>l.active).length}/${islLinks.length}` },
              { label:'Eclipse',     value:`${fleetHealth.inEclipse} sats` },
              { label:'Overrides',   value:`${fleetHealth.overrides}`, color:'#ffaa00' },
              { label:'Model',       value:modelsStatus==='ready'?'ONNX':'MPC', color:modelsStatus==='ready'?'#00f5ff':'#ffaa00' },
            ].map(item => (
              <div key={item.label} style={{
                padding:'6px 8px', borderRadius:'7px',
                background:'rgba(255,255,255,0.02)',
                border:'1px solid rgba(255,255,255,0.05)',
              }}>
                <div style={{ color:'#6b7280', fontSize:'0.58rem', marginBottom:'2px', textTransform:'uppercase', letterSpacing:'1px' }}>{item.label}</div>
                <div style={{ fontWeight:700, fontFamily:"'JetBrains Mono',monospace", color:item.color||'#e8eaf0', fontSize:'0.82rem' }}>{item.value}</div>
              </div>
            ))}
          </div>
        </div>
      </aside>

      {/* ── CENTER (main view) ──────────────────────────────────────────────── */}
      <main className="layout-center glass" style={{ padding:0, overflow:'hidden' }}>
        {screen === 'ORBITAL' && (
          <CelestialView
            sats={sats}
            simTime={simTime}
            selectedSatId={selectedSat}
            onSelect={setSelectedSat}
          />
        )}
        {screen === 'BRAIN' && (
          <div style={{ padding:'20px', height:'100%', overflowY:'auto' }}>
            <AgentBrain sat={selectedSatObj} decision={selectedDecision} />
          </div>
        )}
        {screen === 'TELEMETRY' && (
          <div style={{ padding:'14px', height:'100%', overflow:'auto' }}>
            <TelemetryPanel
              sats={sats}
              selectedId={selectedSat}
              onSelect={setSelectedSat}
              socHistories={socHistories}
              gridMode
            />
          </div>
        )}
        {screen === 'COMPARISON' && (
          <div style={{ padding:'16px', height:'100%', overflowY:'auto' }}>
            <ComparisonPanel sats={sats} />
          </div>
        )}
        {screen === 'DUEL' && <DuelView />}

        {/* Speed/control overlay for orbital view */}
        {screen === 'ORBITAL' && (
          <div style={{
            position:'absolute', bottom:'16px', left:'50%', transform:'translateX(-50%)',
            display:'flex', gap:'8px', zIndex:10,
          }}>
            <div className="glass" style={{ padding:'6px 12px', display:'flex', gap:'8px', alignItems:'center' }}>
              <span style={{ fontSize:'0.65rem', color:'#6b7280', fontFamily:"'JetBrains Mono',monospace" }}>SIM SPEED</span>
              {[1,5,20].map(sp => (
                <button key={sp}
                  className={`btn btn-ghost ${simSpeed===sp ? 'active':''}`}
                  style={{ padding:'3px 10px', fontSize:'0.65rem' }}
                  onClick={() => setSimSpeed(sp)}
                >
                  {sp}×
                </button>
              ))}
            </div>
          </div>
        )}
      </main>

      {/* ── RIGHT SIDEBAR ──────────────────────────────────────────────────── */}
      <aside className="layout-right">
        {/* Agent Brain (compact) */}
        <div className="glass glass-scan" style={{ flex:1, padding:'14px', overflow:'hidden', display:'flex', flexDirection:'column' }}>
          <div className="widget-title holo">
            <Cpu size={13} /> Agent Brain • {selectedSat}
          </div>
          <div style={{ flex:1, overflow:'auto' }}>
            <AgentBrain sat={selectedSatObj} decision={selectedDecision} compact />
          </div>
        </div>

        {/* Anomaly injection panel */}
        <div className="glass" style={{ padding:'14px', flexShrink:0 }}>
          <div className="widget-title">
            <ShieldAlert size={13} /> Anomaly Injection
          </div>
          <div style={{ display:'flex', flexDirection:'column', gap:'6px' }}>
            <button className="btn btn-warn" style={{ width:'100%', justifyContent:'center', fontSize:'0.72rem' }}
              onClick={() => handleInjectAnomaly('solar_fault')}>
              ☀ Solar Fault (−60%)
            </button>
            <button className="btn btn-danger" style={{ width:'100%', justifyContent:'center', fontSize:'0.72rem' }}
              onClick={() => handleInjectAnomaly('battery_drain')}>
              🔋 Battery Drain (−40%)
            </button>
            <button className="btn btn-danger" style={{ width:'100%', justifyContent:'center', fontSize:'0.72rem' }}
              onClick={() => handleInjectAnomaly('thermal_spike')}>
              🌡 Thermal Spike (+30°C)
            </button>
            <button className="btn btn-ghost" style={{ width:'100%', justifyContent:'center', fontSize:'0.72rem' }}
              onClick={() => handleInjectAnomaly('clear')}>
              ✓ Clear Anomaly
            </button>
          </div>
        </div>

        {/* Layer architecture strip */}
        <div className="glass" style={{ padding:'12px', flexShrink:0 }}>
          <div className="widget-title">Architecture Layers</div>
          <div style={{ display:'flex', flexDirection:'column', gap:'3px' }}>
            {[
              { label:'L1: Digital Twin',       color:'#6b7280', active:true },
              { label:'L2: World Model (fψ)',    color:'#00f5ff', active:modelsStatus==='ready' },
              { label:'L2: MAPPO Policy (πθ)',   color:'#7b61ff', active:modelsStatus==='ready' },
              { label:'L2: Safety Gate',         color:'#ffaa00', active:true },
              { label:'L3: Cooperative MARL',    color:'#00ff88', active:true },
              { label:'L4: Coordinator',         color:'#f59e0b', active:true },
              { label:'L5: Anomaly Recovery',    color:'#ff3d3d', active:fleetHealth.critical > 0 },
            ].map(layer => (
              <div key={layer.label} style={{
                display:'flex', alignItems:'center', gap:'8px',
                padding:'4px 8px', borderRadius:'5px',
                background: layer.active ? `${layer.color}0a` : 'transparent',
                border:`1px solid ${layer.active ? `${layer.color}22`: 'transparent'}`,
                fontSize:'0.62rem',
              }}>
                <div style={{
                  width:6, height:6, borderRadius:'50%',
                  background: layer.active ? layer.color : '#374151',
                  boxShadow: layer.active ? `0 0 6px ${layer.color}` : 'none',
                  flexShrink:0,
                }} />
                <span style={{ color: layer.active ? '#e8eaf0' : '#4b5563' }}>{layer.label}</span>
              </div>
            ))}
          </div>
        </div>
      </aside>

      {/* ── BOTTOM LOG ─────────────────────────────────────────────────────── */}
      <footer className="layout-bottom glass" style={{ padding:'8px 16px', overflow:'hidden' }}>
        <div style={{ display:'flex', alignItems:'center', gap:'16px', height:'100%', overflow:'hidden' }}>
          <div style={{ fontWeight:700, color:'var(--holo-primary)', fontSize:'0.68rem',
            fontFamily:"'JetBrains Mono',monospace", borderRight:'1px solid rgba(255,255,255,0.08)',
            paddingRight:'16px', minWidth:'100px', letterSpacing:'1px' }}>
            MISSION LOG
          </div>
          <div style={{ display:'flex', flexDirection:'column', gap:'3px', flex:1, overflow:'hidden' }}>
            {eventLog.slice(0,5).map((log, i) => {
              const colors = { info:'#6b7280', warn:'#ffaa00', success:'#00ff88', critical:'#ff3d3d' };
              return (
                <div key={i} className="log-entry" style={{ color: i===0 ? '#e8eaf0':'#4b5563' }}>
                  <span style={{ color:'var(--holo-primary)', minWidth:'60px' }}>
                    [{(log.t/3600).toFixed(2)}h]
                  </span>
                  <span style={{ color: colors[log.level]||'#6b7280', minWidth:'6px' }}>
                    {log.level==='warn'?'⚠':log.level==='critical'?'🔴':log.level==='success'?'✓':'·'}
                  </span>
                  <span>{log.msg}</span>
                </div>
              );
            })}
          </div>
        </div>
      </footer>
    </div>
  );
}
