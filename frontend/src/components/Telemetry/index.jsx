
/**
 * Telemetry Panel — per-satellite health monitoring.
 * Shows SoC, SoH, Temp, Mode with live sparklines.
 */
import React, { useMemo } from 'react';
import { Battery, Thermometer, Cpu, Radio, Zap, ShieldCheck } from 'lucide-react';
import { LineChart, Line, ResponsiveContainer } from 'recharts';

const MODE_COLORS = {
  PAYLOAD:'#00ff88', RELAY:'#7b61ff', CHARGE:'#f5d800',
  HIBERNATE:'#ff3d3d', PAYLOAD_OFF:'#6b7280',
};
const MODE_ICONS = { PAYLOAD:'⚡', RELAY:'📡', CHARGE:'🔋', HIBERNATE:'💤', PAYLOAD_OFF:'⬛' };

const SoCArc = ({ value, size = 48 }) => {
  const r = 18;
  const circ = 2 * Math.PI * r;
  const offset = circ - (Math.min(100, Math.max(0, value)) / 100) * circ;
  const color  = value < 10 ? '#ff3d3d' : value < 20 ? '#ffaa00' : '#00f5ff';

  return (
    <svg width={size} height={size} viewBox="0 0 44 44" style={{ flexShrink:0 }}>
      <circle cx="22" cy="22" r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="4" />
      <circle
        cx="22" cy="22" r={r}
        fill="none"
        stroke={color}
        strokeWidth="4"
        strokeDasharray={circ}
        strokeDashoffset={offset}
        strokeLinecap="round"
        transform="rotate(-90 22 22)"
        style={{ transition:'stroke-dashoffset 0.6s ease, stroke 0.3s' }}
        filter={`drop-shadow(0 0 4px ${color})`}
      />
      <text x="22" y="26" textAnchor="middle" fill={color} fontSize="9" fontFamily="'JetBrains Mono'" fontWeight="700">
        {value.toFixed(0)}%
      </text>
    </svg>
  );
};

const SatCard = ({ sat, isSelected, onSelect, socHistory }) => {
  const modeColor = MODE_COLORS[sat.mode] || '#6b7280';
  const sparkData = (socHistory || []).slice(-20).map(v => ({ v }));

  return (
    <div
      onClick={() => onSelect(sat.id)}
      style={{
        padding:'10px 12px',
        borderRadius:'10px',
        cursor:'pointer',
        background: isSelected ? `${modeColor}0a` : 'rgba(255,255,255,0.02)',
        border:`1px solid ${isSelected ? `${modeColor}40` : 'rgba(255,255,255,0.05)'}`,
        transition:'all 0.25s',
        display:'flex', flexDirection:'column', gap:'8px',
      }}
    >
      {/* Header row */}
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
        <div style={{ display:'flex', alignItems:'center', gap:'7px' }}>
          <div style={{ width:'7px', height:'7px', borderRadius:'50%', background:sat.color, boxShadow:`0 0 6px ${sat.color}` }} />
          <span style={{ fontWeight:700, fontSize:'0.78rem', fontFamily:"'JetBrains Mono',monospace" }}>{sat.id}</span>
          {sat.eclipseFlag && <span style={{ fontSize:'0.65rem', color:'#6b7280' }}>🌑</span>}
          {sat.anomaly && <span style={{ fontSize:'0.65rem', color:'#ff3d3d', animation:'blink 1s infinite' }}>⚠</span>}
        </div>
        <div style={{
          display:'flex', alignItems:'center', gap:'4px',
          padding:'2px 8px', borderRadius:'4px',
          background:`${modeColor}15`, color:modeColor,
          fontSize:'0.6rem', fontFamily:"'JetBrains Mono',monospace", fontWeight:700,
          border:`1px solid ${modeColor}30`,
        }}>
          {MODE_ICONS[sat.mode]} {sat.mode}
        </div>
      </div>

      {/* Main telemetry row */}
      <div style={{ display:'flex', alignItems:'center', gap:'10px' }}>
        <SoCArc value={sat.soc} size={44} />
        <div style={{ flex:1, display:'flex', flexDirection:'column', gap:'5px' }}>
          {/* SoH */}
          <div>
            <div style={{ display:'flex', justifyContent:'space-between', fontSize:'0.62rem', color:'#6b7280', marginBottom:'2px' }}>
              <span>SoH (Health)</span>
              <span style={{ color:'#00ff88' }}>{((sat.soh||1)*100).toFixed(1)}%</span>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{
                width:`${(sat.soh||1)*100}%`,
                background:`linear-gradient(90deg, #00ff8866, #00ff88)`,
              }} />
            </div>
          </div>
          {/* Temp */}
          <div>
            <div style={{ display:'flex', justifyContent:'space-between', fontSize:'0.62rem', color:'#6b7280', marginBottom:'2px' }}>
              <span>Temperature</span>
              <span style={{ color: sat.temp > 55 ? '#ff3d3d' : sat.temp > 40 ? '#ffaa00' : '#e8eaf0' }}>
                {sat.temp.toFixed(1)}°C
              </span>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{
                width:`${Math.min(100, (sat.temp + 20) / 100 * 100)}%`,
                background: sat.temp > 55
                  ? `linear-gradient(90deg, #ff3d3d44, #ff3d3d)`
                  : sat.temp > 40
                  ? `linear-gradient(90deg, #ffaa0044, #ffaa00)`
                  : `linear-gradient(90deg, #00ff8844, #00f5ff)`
              }} />
            </div>
          </div>
        </div>

        {/* Sparkline */}
        {sparkData.length > 3 && (
          <div style={{ width:'50px', height:'40px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={sparkData}>
                <Line type="monotone" dataKey="v" stroke={modeColor} dot={false} strokeWidth={1.5} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Status */}
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
        <span className={`badge badge-${(sat.status||'NOMINAL').toLowerCase()}`} style={{ fontSize:'0.59rem' }}>
          {sat.status || 'NOMINAL'}
        </span>
        {sat.overriddenThisStep && (
          <span style={{ fontSize:'0.6rem', color:'#ffaa00', display:'flex', alignItems:'center', gap:'3px' }}>
            🛡 Override
          </span>
        )}
      </div>
    </div>
  );
};

const TelemetryPanel = ({ sats, selectedId, onSelect, socHistories, gridMode }) => {
  const fleetStats = useMemo(() => {
    const nominal  = sats.filter(s => s.status === 'NOMINAL').length;
    const warning  = sats.filter(s => s.status === 'WARNING').length;
    const critical = sats.filter(s => s.status === 'CRITICAL').length;
    const avgSoc   = sats.reduce((s, sat) => s + sat.soc, 0) / sats.length;
    const avgSoh   = sats.reduce((s, sat) => s + (sat.soh||1), 0) / sats.length;
    return { nominal, warning, critical, avgSoc, avgSoh };
  }, [sats]);

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:'10px', height:'100%', overflow:'hidden' }}>
      {/* Fleet Summary */}
      {!gridMode && (
        <div style={{
          display:'grid', gridTemplateColumns:'1fr 1fr 1fr',
          gap:'6px', fontSize:'0.65rem',
        }}>
          {[
            { label:'ONLINE', value:fleetStats.nominal, color:'#00ff88' },
            { label:'WARN',   value:fleetStats.warning, color:'#ffaa00' },
            { label:'CRIT',   value:fleetStats.critical,color:'#ff3d3d' },
          ].map(m => (
            <div key={m.label} style={{
              textAlign:'center', padding:'6px 4px',
              background:'rgba(255,255,255,0.02)',
              border:`1px solid ${m.value > 0 && m.label !== 'ONLINE' ? `${m.color}40` : 'rgba(255,255,255,0.05)'}`,
              borderRadius:'7px',
            }}>
              <div style={{ color:m.color, fontWeight:700, fontSize:'1.1rem', fontFamily:"'JetBrains Mono',monospace" }}>{m.value}</div>
              <div style={{ color:'#6b7280', letterSpacing:'1px' }}>{m.label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Avg bars */}
      {!gridMode && (
        <div style={{ fontSize:'0.65rem', color:'#6b7280' }}>
          <div style={{ display:'flex', justifyContent:'space-between', marginBottom:'4px' }}>
            <span>Fleet Avg SoC</span>
            <span style={{ color:'#00f5ff' }}>{fleetStats.avgSoc.toFixed(1)}%</span>
          </div>
          <div className="progress-track" style={{ marginBottom:'8px' }}>
            <div className="progress-fill" style={{
              width:`${fleetStats.avgSoc}%`,
              background:'linear-gradient(90deg,#00f5ff44,#00f5ff)',
            }} />
          </div>
          <div style={{ display:'flex', justifyContent:'space-between', marginBottom:'4px' }}>
            <span>Fleet Avg SoH</span>
            <span style={{ color:'#00ff88' }}>{(fleetStats.avgSoh*100).toFixed(1)}%</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{
              width:`${fleetStats.avgSoh*100}%`,
              background:'linear-gradient(90deg,#00ff8844,#00ff88)',
            }} />
          </div>
        </div>
      )}

      {/* Sat list */}
      <div style={{
        flex:1,
        overflowY:'auto',
        display: gridMode ? 'grid' : 'flex',
        flexDirection: gridMode ? undefined : 'column',
        gridTemplateColumns: gridMode ? 'repeat(auto-fill,minmax(270px,1fr))' : undefined,
        gap:'6px',
        paddingRight:'2px',
      }}>
        {sats.map(sat => (
          <SatCard
            key={sat.id}
            sat={sat}
            isSelected={sat.id === selectedId}
            onSelect={onSelect}
            socHistory={(socHistories || {})[sat.id]}
          />
        ))}
      </div>
    </div>
  );
};

export default TelemetryPanel;
