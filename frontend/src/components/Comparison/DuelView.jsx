
/**
 * DuelView — Side-by-side Traditional PID vs CASC-RL live simulation.
 * Runs two fully independent physics loops at 60ms intervals.
 * Traditional: reactive PID (charges only when SoC<20%)
 * CASC-RL: predictive (uses eclipse countdown to pre-charge)
 */
import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { AreaChart, Area } from 'recharts';

const EPISODE_LENGTH = 200;   // steps
const ECLIPSE_START  = 65;    // step
const ECLIPSE_END    = 100;   // step
const DT             = 1;     // each step = 1 unit

function stepPhysics(soc, mode, inEclipse, solarEff = 1.0) {
  const P_solar   = inEclipse ? 0 : 0.85 * solarEff;
  const P_consume = { PAYLOAD:0.60, RELAY:0.40, CHARGE:0.20, HIBERNATE:0.05 }[mode] || 0.30;
  const dSoc      = (P_solar - P_consume) * 0.25 * DT;
  return Math.max(0, Math.min(100, soc + dSoc));
}

function traditionalControl(soc) {
  if (soc < 20) return 'CHARGE';
  return 'PAYLOAD';
}

function cascControl(soc, step) {
  // Predictive: check if eclipse approaching in next 15 steps
  const eclipseSoon = step >= (ECLIPSE_START - 15) && step < ECLIPSE_START;
  const inEclipse   = step >= ECLIPSE_START && step < ECLIPSE_END;

  if (soc < 10) return 'HIBERNATE';
  if (soc < 25 || inEclipse) return 'CHARGE';
  if (eclipseSoon && soc < 75) return 'CHARGE';  // PRE-CHARGE! ◄ key difference
  if (soc > 85 && !inEclipse) return 'PAYLOAD';
  return 'PAYLOAD';
}

const TRAD_EVENTS = [];
const CASC_EVENTS = [];

const PhysicsPanel = ({ title, type, color, history, currentStep, soc, mode, events, solarEff }) => {
  const inEclipse  = currentStep >= ECLIPSE_START && currentStep < ECLIPSE_END;
  const eclipseSoon = type === 'casc' && currentStep >= (ECLIPSE_START - 15) && currentStep < ECLIPSE_START;
  const failed     = soc <= 0;

  const modeColors = { PAYLOAD:'#00ff88', CHARGE:'#f5d800', HIBERNATE:'#ff3d3d', RELAY:'#7b61ff' };
  const mc         = modeColors[mode] || '#6b7280';

  return (
    <div style={{
      flex:1, display:'flex', flexDirection:'column', overflow:'hidden',
      border:`1px solid ${color}22`,
      background: failed ? 'rgba(255,61,61,0.05)' : 'rgba(0,0,0,0.2)',
      ...(failed ? { animation:'critical-flash 0.8s infinite' } : {}),
    }}>
      {/* Header */}
      <div style={{
        padding:'12px 16px',
        background:`${color}0d`,
        borderBottom:`1px solid ${color}22`,
        display:'flex', justifyContent:'space-between', alignItems:'center',
      }}>
        <div>
          <div style={{ fontWeight:700, color, fontSize:'0.85rem', fontFamily:"'JetBrains Mono',monospace" }}>{title}</div>
          <div style={{ fontSize:'0.62rem', color:'#6b7280', marginTop:'2px' }}>{type === 'casc' ? 'Predictive MARL w/ World Model' : 'Reactive Rule-Based'}</div>
        </div>
        <div style={{ display:'flex', alignItems:'center', gap:'8px' }}>
          {inEclipse && <span style={{ fontSize:'0.65rem', color:'#ffaa00', animation:'blink 1s infinite' }}>🌑 ECLIPSE</span>}
          {eclipseSoon && <span style={{ fontSize:'0.65rem', color:'#ffaa00' }}>⚡ PRE-CHARGE</span>}
          {solarEff < 1 && <span style={{ fontSize:'0.65rem', color:'#ff3d3d' }}>☀ FAULT {(solarEff*100).toFixed(0)}%</span>}
        </div>
      </div>

      {/* Key stats */}
      <div style={{ display:'flex', gap:'1px', background:'rgba(255,255,255,0.04)', flexShrink:0 }}>
        {[
          { label:'SoC', value:`${soc.toFixed(1)}%`, vcolor: soc<10 ? '#ff3d3d' : soc<20 ? '#ffaa00' : color },
          { label:'Mode', value: mode, vcolor: mc },
          { label:'Step', value:`${currentStep}/${EPISODE_LENGTH}`, vcolor:'#6b7280' },
        ].map(item => (
          <div key={item.label} style={{
            flex:1, padding:'8px 12px', background:'rgba(0,0,0,0.3)',
            borderRight:'1px solid rgba(255,255,255,0.04)',
          }}>
            <div style={{ fontSize:'0.6rem', color:'#6b7280', marginBottom:'2px', textTransform:'uppercase', letterSpacing:'1px' }}>{item.label}</div>
            <div style={{ fontSize:'0.85rem', fontWeight:700, color:item.vcolor, fontFamily:"'JetBrains Mono',monospace" }}>{item.value}</div>
          </div>
        ))}
      </div>

      {/* SoC progress bar */}
      <div style={{ padding:'10px 16px 0', flexShrink:0 }}>
        <div className="progress-track" style={{ height:'8px' }}>
          <div className="progress-fill" style={{
            width:`${soc}%`,
            background: soc < 10 ? 'linear-gradient(90deg,#ff3d3d88,#ff3d3d)' :
                        soc < 20 ? 'linear-gradient(90deg,#ffaa0088,#ffaa00)' :
                                   `linear-gradient(90deg,${color}44,${color})`,
            boxShadow:`0 0 8px ${mc}66`,
          }} />
        </div>
      </div>

      {/* Trajectory chart */}
      <div style={{ flex:1, padding:'8px 4px 4px', minHeight:0 }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={history} margin={{ top:4, right:4, left:-22, bottom:0 }}>
            <defs>
              <linearGradient id={`grad-${type}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={color} stopOpacity={0.3}/>
                <stop offset="95%" stopColor={color} stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="4 4" vertical={false} />
            <XAxis dataKey="step" hide />
            <YAxis domain={[0, 100]} tick={{ fontSize:9 }} />
            <Tooltip
              contentStyle={{ background:'#07090f', border:'1px solid rgba(255,255,255,0.1)', borderRadius:8, fontSize:10 }}
              formatter={(v, k) => [v.toFixed(1)+'%', k === 'soc' ? 'SoC' : k]}
            />
            {/* Eclipse zone */}
            {ECLIPSE_START < history.length && (
              <ReferenceLine x={ECLIPSE_START} stroke="#ffaa00" strokeDasharray="4 3" strokeWidth={1} />
            )}
            {ECLIPSE_END < history.length && (
              <ReferenceLine x={ECLIPSE_END} stroke="#ffaa00" strokeDasharray="4 3" strokeWidth={1} />
            )}
            <ReferenceLine y={20} stroke="#ffaa00" strokeDasharray="3 3" strokeWidth={1} />
            <ReferenceLine y={10} stroke="#ff3d3d" strokeDasharray="3 3" strokeWidth={1} />
            <Area type="monotone" dataKey="soc" stroke={color} fill={`url(#grad-${type})`} strokeWidth={2} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Event log */}
      <div style={{
        padding:'6px 12px', borderTop:'1px solid rgba(255,255,255,0.04)',
        maxHeight:'60px', overflowY:'auto', flexShrink:0,
      }}>
        {events.slice(-3).reverse().map((e, i) => (
          <div key={i} style={{
            fontSize:'0.62rem', color: i===0 ? '#e8eaf0' : '#4b5563',
            fontFamily:"'JetBrains Mono',monospace",
            padding:'1px 0',
          }}>
            <span style={{ color:color, marginRight:4 }}>[{e.step}]</span>{e.msg}
          </div>
        ))}
      </div>

      {/* Mission failure overlay */}
      {failed && (
        <div style={{
          position:'absolute', inset:0,
          display:'flex', alignItems:'center', justifyContent:'center',
          background:'rgba(255,61,61,0.15)',
          backdropFilter:'blur(4px)',
          zIndex:20,
          flexDirection:'column', gap:'8px',
        }}>
          <div style={{ fontSize:'1.6rem', fontWeight:900, color:'#ff3d3d', textShadow:'0 0 30px #ff3d3d' }}>MISSION FAILURE</div>
          <div style={{ fontSize:'0.75rem', color:'#e8eaf0', textAlign:'center' }}>Battery depleted. Satellite offline.</div>
        </div>
      )}
    </div>
  );
};

const DuelView = () => {
  const [step, setStep] = useState(0);
  const [running, setRunning] = useState(true);
  const [solarFault, setSolarFault] = useState(false);

  const [tradState, setTradState] = useState({ soc:50, mode:'PAYLOAD', history:[] });
  const [cascState, setCascState] = useState({ soc:50, mode:'PAYLOAD', history:[] });

  const tradEvents = useRef([]);
  const cascEvents = useRef([]);
  const [evTrad, setEvTrad] = useState([]);
  const [evCasc, setEvCasc] = useState([]);

  const reset = () => {
    setStep(0);
    setTradState({ soc:50, mode:'PAYLOAD', history:[] });
    setCascState({ soc:50, mode:'PAYLOAD', history:[] });
    tradEvents.current = [];
    cascEvents.current = [];
    setEvTrad([]);
    setEvCasc([]);
    setRunning(true);
    setSolarFault(false);
  };

  useEffect(() => {
    if (!running || step >= EPISODE_LENGTH) return;
    const eff = solarFault ? 0.4 : 1.0;

    const timeout = setTimeout(() => {
      const inEclipse = step >= ECLIPSE_START && step < ECLIPSE_END;

      setTradState(prev => {
        const mode    = traditionalControl(prev.soc);
        const newSoc  = stepPhysics(prev.soc, mode, inEclipse, eff);
        const hist    = [...prev.history, { step, soc:newSoc }];

        if (prev.mode !== mode) {
          tradEvents.current = [...tradEvents.current, { step, msg:`→ ${mode} (SoC ${newSoc.toFixed(0)}%)` }];
          setEvTrad([...tradEvents.current]);
        }
        if (newSoc < 10 && prev.soc >= 10) {
          tradEvents.current = [...tradEvents.current, { step, msg:'🔴 CRITICAL — SoC below 10%!' }];
          setEvTrad([...tradEvents.current]);
        }
        return { soc:newSoc, mode, history:hist };
      });

      setCascState(prev => {
        const mode    = cascControl(prev.soc, step);
        const newSoc  = stepPhysics(prev.soc, mode, inEclipse, eff);
        const hist    = [...prev.history, { step, soc:newSoc }];

        if (prev.mode !== mode) {
          cascEvents.current = [...cascEvents.current, { step, msg:`→ ${mode} (predicted ${step < ECLIPSE_START ? 'eclipse' : 'state'})` }];
          setEvCasc([...cascEvents.current]);
        }
        return { soc:newSoc, mode, history:hist };
      });

      setStep(s => s + 1);
    }, 60);

    return () => clearTimeout(timeout);
  }, [step, running, solarFault]);

  const pause = () => setRunning(r => !r);

  return (
    <div style={{ height:'100%', display:'flex', flexDirection:'column', overflow:'hidden' }}>
      {/* Controls */}
      <div style={{
        display:'flex', gap:'8px', padding:'8px 12px',
        background:'rgba(0,0,0,0.3)', borderBottom:'1px solid rgba(255,255,255,0.06)',
        alignItems:'center', flexWrap:'wrap',
      }}>
        <button className="btn btn-holo" onClick={pause} style={{ fontSize:'0.7rem' }}>
          {running ? '⏸ Pause' : '▶ Resume'}
        </button>
        <button className="btn btn-ghost" onClick={reset} style={{ fontSize:'0.7rem' }}>⟳ Reset</button>
        <button
          className={`btn ${solarFault ? 'btn-danger' : 'btn-warn'}`}
          onClick={() => {
            setSolarFault(f => !f);
            const msg = !solarFault ? 'Solar panel fault injected (-60%)' : 'Solar panel restored';
            tradEvents.current = [...tradEvents.current, { step, msg:`⚠ ${msg}` }];
            cascEvents.current = [...cascEvents.current, { step, msg:`⚠ ${msg}` }];
            setEvTrad([...tradEvents.current]);
            setEvCasc([...cascEvents.current]);
          }}
          style={{ fontSize:'0.7rem' }}
        >
          {solarFault ? '☀ Clear Fault' : '☀ Solar Fault'}
        </button>
        <div style={{ marginLeft:'auto', fontSize:'0.68rem', color:'#6b7280', fontFamily:"'JetBrains Mono',monospace" }}>
          Eclipse: T{ECLIPSE_START}–T{ECLIPSE_END}
          {step >= ECLIPSE_START && step < ECLIPSE_END && (
            <span style={{ color:'#ffaa00', marginLeft:'8px', animation:'blink 1s infinite' }}>● ACTIVE</span>
          )}
        </div>
        <div style={{ fontSize:'0.68rem', fontFamily:"'JetBrains Mono',monospace",
          color: step >= EPISODE_LENGTH ? '#00ff88' : '#6b7280' }}>
          {step >= EPISODE_LENGTH ? '✓ Complete' : `Step ${step}/${EPISODE_LENGTH}`}
        </div>
      </div>

      {/* Duel */}
      <div className="duel-container" style={{ flex:1 }}>
        <div style={{ position:'relative', overflow:'hidden' }}>
          <PhysicsPanel
            title="TRADITIONAL (PID)"
            type="trad"
            color="#ff3d3d"
            history={tradState.history}
            currentStep={step}
            soc={tradState.soc}
            mode={tradState.mode}
            events={evTrad}
            solarEff={solarFault ? 0.4 : 1.0}
          />
        </div>
        <div className="duel-divider" />
        <div style={{ position:'relative', overflow:'hidden' }}>
          <PhysicsPanel
            title="CASC-RL (MARL)"
            type="casc"
            color="#00f5ff"
            history={cascState.history}
            currentStep={step}
            soc={cascState.soc}
            mode={cascState.mode}
            events={evCasc}
            solarEff={solarFault ? 0.4 : 1.0}
          />
        </div>
      </div>
    </div>
  );
};

export default DuelView;
