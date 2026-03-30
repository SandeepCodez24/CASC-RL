
/**
 * AgentBrain — visualizes the internal cognitive loop of a satellite agent.
 * Shows: World Model forecast, MPC action scoring, safety state, decision trail.
 * Data comes directly from cognitiveDecision() in simulation.js.
 */
import React, { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
  ResponsiveContainer, AreaChart, Area, BarChart, Bar, Cell
} from 'recharts';
import { Brain, Zap, ShieldCheck, ShieldAlert, AlertTriangle, ChevronRight, CheckCircle, XCircle, Clock } from 'lucide-react';

const ACTION_NAMES = ['PAYLOAD_ON','PAYLOAD_OFF','HIBERNATE','RELAY_MODE','CHARGE_PRIORITY'];
const ACTION_COLORS = ['#00ff88','#6b7280','#ff3d3d','#7b61ff','#f5d800'];
const ACTION_ICONS  = ['⚡','⬛','💤','📡','🔋'];

const SafetyStateBar = ({ sat, decision }) => {
  const status = sat?.status || 'NOMINAL';
  const colors = { NOMINAL:'#00ff88', WARNING:'#ffaa00', CRITICAL:'#ff3d3d', RECOVERY:'#7b61ff' };
  const color = colors[status] || '#00ff88';

  return (
    <div style={{
      display:'flex', alignItems:'center', justifyContent:'space-between',
      padding:'10px 14px', borderRadius:'8px',
      background:`${color}12`, border:`1px solid ${color}44`,
      ...(status === 'CRITICAL' ? { animation:'critical-flash 0.8s infinite' } : {}),
    }}>
      <div style={{ display:'flex', alignItems:'center', gap:'8px' }}>
        {status === 'NOMINAL'  && <ShieldCheck  size={16} color={color} />}
        {status === 'WARNING'  && <ShieldAlert  size={16} color={color} />}
        {status === 'CRITICAL' && <AlertTriangle size={16} color={color} />}
        {status === 'RECOVERY' && <ShieldCheck  size={16} color={color} />}
        <span style={{ color, fontWeight:700, fontSize:'0.75rem', fontFamily:"'JetBrains Mono',monospace" }}>
          {status}
        </span>
      </div>
      {decision?.overridden && (
        <div style={{ fontSize:'0.65rem', color:'#ffaa00', display:'flex', alignItems:'center', gap:'4px' }}>
          <AlertTriangle size={11} />
          Safety Override: {decision.overrideReason}
        </div>
      )}
      {!decision?.overridden && (
        <span style={{ fontSize:'0.65rem', color:'#6b7280' }}>Policy Accepted</span>
      )}
    </div>
  );
};

const WorldModelForecast = ({ decision, sat }) => {
  if (!decision?.chosenFutures) return null;
  const futures = decision.chosenFutures;

  const chartData = [
    { t: 0, label:'Now', soc: sat?.soc || 0, temp: sat?.temp || 0 },
    ...futures.map((f, i) => ({
      t: i + 1,
      label: `+${(i+1)*60}s`,
      soc:  parseFloat(f.soc.toFixed(1)),
      temp: parseFloat(f.temp.toFixed(1)),
      eclipse: f.eclipseRisk ? 100 : 0,
    }))
  ];

  const ec = decision.eclipseCountdown;

  return (
    <div>
      {/* Eclipse countdown */}
      {ec !== null && (
        <div style={{
          display:'flex', justifyContent:'space-between', alignItems:'center',
          padding:'8px 12px', borderRadius:'7px', marginBottom:'10px',
          background: ec < 120 ? 'rgba(255,170,0,0.1)' : 'rgba(0,245,255,0.06)',
          border:`1px solid ${ec < 120 ? 'rgba(255,170,0,0.3)' : 'rgba(0,245,255,0.15)'}`,
        }}>
          <div style={{ display:'flex', alignItems:'center', gap:'6px', fontSize:'0.72rem' }}>
            <Clock size={13} color={ec < 120 ? '#ffaa00' : '#6b7280'} />
            <span style={{ color:'#6b7280' }}>Eclipse prediction:</span>
          </div>
          <span style={{
            color: ec < 120 ? '#ffaa00' : '#00f5ff',
            fontFamily:"'JetBrains Mono',monospace",
            fontWeight:700, fontSize:'0.75rem',
          }}>
            {ec < 120 ? '⚠ ' : ''}{Math.floor(ec/60)}m {ec % 60}s
          </span>
        </div>
      )}

      {/* SoC forecast chart */}
      <div style={{ height:'120px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top:5, right:5, left:-20, bottom:0 }}>
            <defs>
              <linearGradient id="socGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#00f5ff" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#00f5ff" stopOpacity={0.02} />
              </linearGradient>
              <linearGradient id="eclipseGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ffaa00" stopOpacity={0.12} />
                <stop offset="100%" stopColor="#ffaa00" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="4 4" vertical={false} />
            <XAxis dataKey="label" tick={{ fontSize:9 }} />
            <YAxis domain={[0, 100]} tick={{ fontSize:9 }} />
            <Tooltip
              contentStyle={{ background:'#07090f', border:'1px solid rgba(255,255,255,0.1)', borderRadius:'8px' }}
              labelStyle={{ color:'#6b7280', fontSize:10 }}
              itemStyle={{ fontSize:10 }}
            />
            <Area dataKey="eclipse" fill="url(#eclipseGrad)" stroke="none" name="Eclipse Zone" />
            <Area
              type="monotone" dataKey="soc" stroke="#00f5ff"
              fill="url(#socGrad)" strokeWidth={2} dot={false} name="SoC (%)"
            />
            <ReferenceLine y={15} stroke="#ff3d3d" strokeDasharray="3 3" strokeWidth={1} label={{ value:'SoC_min', position:'right', fontSize:8, fill:'#ff3d3d' }} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Summary row */}
      <div style={{ display:'flex', justifyContent:'space-between', fontSize:'0.68rem', color:'#6b7280', marginTop:'8px' }}>
        <span>Projected SoC at t+5: <b style={{ color:'#00f5ff' }}>{futures[4]?.soc.toFixed(1)}%</b></span>
        <span>Projected Temp: <b style={{ color: futures[4]?.temp > 55 ? '#ff3d3d': '#e8eaf0' }}>{futures[4]?.temp.toFixed(1)}°C</b></span>
      </div>
    </div>
  );
};

const ActionScoring = ({ decision, compact }) => {
  if (!decision) return null;
  const { scores, logits, safeAction, rawAction, overridden } = decision;

  // Normalize scores to 0-1 for display
  const scoreArr = scores || Array(5).fill(0);
  const minS = Math.min(...scoreArr);
  const maxS = Math.max(...scoreArr);
  const normScores = scoreArr.map(s => maxS > minS ? (s - minS) / (maxS - minS) : 0.5);

  // Normalize logits with softmax
  let softmax = null;
  if (logits) {
    const expL = logits.map(l => Math.exp(l - Math.max(...logits)));
    const sumE  = expL.reduce((a,b) => a+b, 0);
    softmax = expL.map(e => e / sumE);
  }

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:'6px' }}>
      {ACTION_NAMES.map((name, i) => {
        const isSafe  = i === safeAction;
        const isRaw   = i === rawAction;
        const rejected = overridden && isRaw && i !== safeAction;
        const color   = isSafe ? ACTION_COLORS[i] : '#6b7280';
        const score   = normScores[i];
        const prob    = softmax ? (softmax[i] * 100) : (score * 100);

        return (
          <div key={name} style={{
            display:'flex', alignItems:'center', gap:'8px',
            padding: compact ? '5px 8px' : '8px 10px',
            borderRadius:'7px',
            background: isSafe ? `${ACTION_COLORS[i]}10` : 'rgba(255,255,255,0.02)',
            border:`1px solid ${isSafe ? `${ACTION_COLORS[i]}30` : 'rgba(255,255,255,0.04)'}`,
            transition:'all 0.3s',
          }}>
            <span style={{ fontSize:'13px', minWidth:'18px' }}>{ACTION_ICONS[i]}</span>
            <div style={{ flex:1, minWidth:0 }}>
              <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'3px' }}>
                <span style={{
                  fontSize:'0.72rem', fontWeight: isSafe ? 700 : 400,
                  color: isSafe ? ACTION_COLORS[i] : '#9ca3af',
                  fontFamily:"'JetBrains Mono',monospace",
                }}>
                  {name}
                </span>
                <div style={{ display:'flex', alignItems:'center', gap:'6px' }}>
                  <span style={{ fontSize:'0.65rem', color:isSafe ? ACTION_COLORS[i] : '#4b5563', fontFamily:"'JetBrains Mono',monospace" }}>
                    {prob.toFixed(1)}%
                  </span>
                  {isSafe && <CheckCircle size={12} color={ACTION_COLORS[i]} />}
                  {rejected && <XCircle size={12} color="#ff3d3d" />}
                </div>
              </div>
              {!compact && (
                <div style={{ display:'flex', gap:'4px', alignItems:'center' }}>
                  <div className="score-bar-track" style={{ flex:1 }}>
                    <div className="score-bar-fill" style={{
                      width:`${score * 100}%`,
                      background: isSafe
                        ? `linear-gradient(90deg, ${ACTION_COLORS[i]}88, ${ACTION_COLORS[i]})`
                        : 'rgba(255,255,255,0.15)',
                    }} />
                  </div>
                  {/* Actor prob bar (if available) */}
                  {softmax && (
                    <div className="score-bar-track" style={{ width:'40px' }}>
                      <div className="score-bar-fill" style={{
                        width:`${softmax[i]*100}%`,
                        background: '#7b61ff88',
                      }} />
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        );
      })}
      {!compact && (
        <div style={{ display:'flex', gap:'12px', marginTop:'4px', fontSize:'0.6rem', color:'#4b5563' }}>
          <span>█ MPC score</span>
          {softmax && <span style={{ color:'#7b61ff88' }}>█ Actor prob</span>}
        </div>
      )}
    </div>
  );
};

const AgentBrain = ({ sat, decision, compact = false }) => {
  if (!sat) return (
    <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:'100%', color:'#6b7280', fontSize:'0.8rem' }}>
      Select a satellite
    </div>
  );

  const status = sat.status || 'NOMINAL';

  return (
    <div style={{ display:'flex', flexDirection:'column', gap: compact ? '10px' : '14px', height:'100%', overflow:'hidden' }}>
      {/* Header */}
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
        <div style={{ display:'flex', alignItems:'center', gap:'7px', fontSize:'0.78rem', fontWeight:600 }}>
          <Brain size={15} color="#00f5ff" />
          <span style={{ fontFamily:"'JetBrains Mono',monospace" }}>{sat.id}</span>
        </div>
        {decision && (
          <div style={{
            fontSize:'0.6rem', color:'#6b7280',
            background:'rgba(0,0,0,0.3)', padding:'2px 8px',
            borderRadius:'4px', fontFamily:"'JetBrains Mono',monospace",
          }}>
            {decision.eclipseFlag ? '🌑 ECLIPSE' : '☀ SUNLIT'}
          </div>
        )}
      </div>

      {/* Current State Row */}
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:'6px' }}>
        {[
          { label:'SoC',  value:`${sat.soc?.toFixed(1)}%`,   color: sat.soc < 15 ? '#ff3d3d' : sat.soc < 25 ? '#ffaa00' : '#00f5ff' },
          { label:'SoH',  value:`${((sat.soh||1)*100).toFixed(1)}%`, color:'#00ff88' },
          { label:'Temp', value:`${sat.temp?.toFixed(1)}°C`,  color: sat.temp > 55 ? '#ff3d3d' : sat.temp > 40 ? '#ffaa00' : '#e8eaf0' },
        ].map(({ label, value, color }) => (
          <div key={label} style={{
            padding:'7px 8px', borderRadius:'7px',
            background:'rgba(255,255,255,0.03)',
            border:'1px solid rgba(255,255,255,0.06)',
            textAlign:'center',
          }}>
            <div style={{ fontSize:'0.6rem', color:'#6b7280', marginBottom:'3px', fontFamily:"'JetBrains Mono',monospace" }}>{label}</div>
            <div style={{ fontSize:'0.85rem', fontWeight:700, color, fontFamily:"'JetBrains Mono',monospace" }}>{value}</div>
          </div>
        ))}
      </div>

      {/* Safety state */}
      <SafetyStateBar sat={sat} decision={decision} />

      {/* World model forecast (hide in compact) */}
      {!compact && decision && (
        <div>
          <div className="widget-title holo">
            <Zap size={13} /> World Model Forecast (k=5)
          </div>
          <WorldModelForecast decision={decision} sat={sat} />
        </div>
      )}

      {/* Action scoring */}
      <div style={{ flex:1, overflow:'hidden' }}>
        <div className="widget-title">
          <ChevronRight size={13} /> Policy Evaluation (πθ)
        </div>
        <div style={{ overflowY:'auto', maxHeight: compact ? '200px' : '300px' }}>
          <ActionScoring decision={decision} compact={compact} />
        </div>
      </div>
    </div>
  );
};

export default AgentBrain;
