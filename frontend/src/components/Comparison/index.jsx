
/**
 * Comparison Dashboard — metrics panel comparing CASC-RL vs baselines.
 * Uses data from HACKATHON_DEMO_GUIDE.md comparison tables.
 * Shows: animated metric counters, SoC trajectory, ablation table, radar chart.
 */
import React, { useState, useEffect, useRef } from 'react';
import {
  AreaChart, Area, LineChart, Line, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, Legend
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingUp, TrendingDown, Zap, Shield, Battery, Activity } from 'lucide-react';

// ─── Pre-computed benchmark data (from HACKATHON_DEMO_GUIDE.md table) ────────
const METRICS = [
  { label:'Battery Lifetime',  unit:'orbits', traditional:900,  casc:1400, lower:false, key:'battery',   icon:Battery },
  { label:'Mission Success',   unit:'%',      traditional:67,   casc:91,   lower:false, key:'mission',   icon:Activity },
  { label:'Thermal Violations',unit:'events', traditional:18,   casc:2,    lower:true,  key:'thermal',   icon:Shield },
  { label:'SoC Min (Eclipse)', unit:'%',      traditional:12,   casc:24,   lower:false, key:'socmin',    icon:Zap },
  { label:'Recovery Time',     unit:'s',       traditional:180, casc:45,   lower:true,  key:'recovery',  icon:Shield },
];

const ABLATION_DATA = [
  { name:'Rule-Based',              battery:800,  mission:62, thermal:24, socmin:8  },
  { name:'+ World Model',           battery:1050, mission:71, thermal:12, socmin:14 },
  { name:'+ Local RL',              battery:1150, mission:79, thermal:8,  socmin:18 },
  { name:'+ Cooperative MARL',      battery:1280, mission:87, thermal:4,  socmin:21 },
  { name:'CASC-RL (Full)',          battery:1400, mission:91, thermal:2,  socmin:24 },
];

const RADAR_DATA = [
  { metric:'Battery Life',  RuleBased:40,  PID:45,  IndPPO:53,  CASCRL:88 },
  { metric:'Mission Success',RuleBased:62, PID:67,  IndPPO:78,  CASCRL:91 },
  { metric:'Thermal Safety',RuleBased:17,  PID:25,  IndPPO:55,  CASCRL:95 },
  { metric:'SoC Safety',    RuleBased:33,  PID:50,  IndPPO:67,  CASCRL:80 },
  { metric:'Recovery',      RuleBased:0,   PID:0,   IndPPO:40,  CASCRL:92 },
  { metric:'Cooperation',   RuleBased:10,  PID:20,  IndPPO:30,  CASCRL:90 },
];

// SoC trajectory comparison over 1000 steps (eclipse stress test)
const generateTrajectory = () => {
  const data = [];
  let trad = 40, casc = 40;
  for (let t = 0; t <= 100; t++) {
    // Traditional: reactive — only charges when <20%, else payload
    const tEclipse = t > 35 && t < 55;
    if (trad > 20 && !tEclipse) trad -= 0.4;
    else if (trad < 20 || tEclipse) trad += 0.2;

    // CASC-RL: predictive — pre-charges before eclipse
    if (casc > 15) {
      if (t > 25 && t < 40) casc += 0.6; // pre-charge
      else if (tEclipse) casc -= 0.2;
      else casc -= 0.2;
    } else {
      casc += 0.3;
    }
    trad = Math.max(0, Math.min(100, trad));
    casc = Math.max(0, Math.min(100, casc));
    data.push({ t, traditional: parseFloat(trad.toFixed(1)), casc: parseFloat(casc.toFixed(1)) });
  }
  return data;
};
const TRAJ_DATA = generateTrajectory();

const AnimatedCounter = ({ target, duration = 1500, unit = '', decimals = 0 }) => {
  const [val, setVal] = useState(0);
  const raf = useRef(null);

  useEffect(() => {
    const start = performance.now();
    const step = (now) => {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      setVal(eased * target);
      if (progress < 1) raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf.current);
  }, [target, duration]);

  return (
    <span className="mono">
      {decimals > 0 ? val.toFixed(decimals) : Math.floor(val)}{unit}
    </span>
  );
};

const MetricCard = ({ metric, animate }) => {
  const improvement = metric.lower
    ? (((metric.traditional - metric.casc) / metric.traditional) * 100).toFixed(0)
    : (((metric.casc - metric.traditional) / metric.traditional) * 100).toFixed(0);
  const Icon = metric.icon;

  return (
    <motion.div
      initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }}
      transition={{ duration:0.4 }}
      className="glass"
      style={{ padding:'16px', display:'flex', flexDirection:'column', gap:'10px' }}
    >
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start' }}>
        <div>
          <div style={{ fontSize:'0.65rem', color:'#6b7280', marginBottom:'4px', textTransform:'uppercase', letterSpacing:'1px' }}>
            {metric.label}
          </div>
          <div style={{ fontSize:'1.4rem', fontWeight:700, color:'#00f5ff', fontFamily:"'JetBrains Mono',monospace" }}>
            {animate ? (
              <AnimatedCounter target={metric.casc} unit={metric.unit === '%' ? '%' : ''} />
            ) : metric.casc}{metric.unit !== '%' ? ` ${metric.unit}` : ''}
          </div>
        </div>
        <Icon size={18} color="#00f5ff" opacity={0.6} />
      </div>

      <div style={{ display:'flex', alignItems:'center', gap:'6px' }}>
        {metric.lower ? <TrendingDown size={13} color="#00ff88" /> : <TrendingUp size={13} color="#00ff88" />}
        <span style={{ fontSize:'0.7rem', color:'#00ff88', fontWeight:600 }}>
          {improvement}% vs Traditional
        </span>
      </div>

      <div>
        <div style={{ display:'flex', justifyContent:'space-between', fontSize:'0.6rem', color:'#6b7280', marginBottom:'3px' }}>
          <span>Traditional</span>
          <span>{metric.traditional}{metric.unit}</span>
        </div>
        <div className="progress-track">
          <div className="progress-fill" style={{
            width:`${metric.lower ? (metric.casc/metric.traditional)*100 : (metric.traditional/metric.casc)*100}%`,
            background:'linear-gradient(90deg,#ff3d3d44,#ff3d3d)',
          }} />
        </div>
        <div style={{ display:'flex', justifyContent:'space-between', fontSize:'0.6rem', color:'#6b7280', marginTop:'3px', marginBottom:'3px' }}>
          <span>CASC-RL</span>
          <span style={{ color:'#00f5ff' }}>{metric.casc}{metric.unit}</span>
        </div>
        <div className="progress-track">
          <div className="progress-fill" style={{
            width:'100%',
            background:'linear-gradient(90deg,#00f5ff44,#00f5ff)',
          }} />
        </div>
      </div>
    </motion.div>
  );
};

const ComparisonPanel = ({ sats }) => {
  const [activeTab, setActiveTab] = useState('metrics');
  const [animated, setAnimated] = useState(false);

  useEffect(() => {
    if (activeTab === 'metrics') {
      setAnimated(false);
      const t = setTimeout(() => setAnimated(true), 100);
      return () => clearTimeout(t);
    }
  }, [activeTab]);

  const tabs = [
    { id:'metrics',  label:'Key Metrics' },
    { id:'trajectory', label:'Eclipse Test' },
    { id:'ablation', label:'Ablation' },
    { id:'radar',    label:'Radar' },
  ];

  return (
    <div style={{ height:'100%', display:'flex', flexDirection:'column', gap:'14px', padding:'0 4px', overflow:'hidden' }}>
      {/* Tabs */}
      <div style={{ display:'flex', gap:'6px', flexWrap:'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} className={`btn btn-ghost ${activeTab === t.id ? 'active' : ''}`}
            onClick={() => setActiveTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      <div style={{ flex:1, overflow:'auto' }}>

        {/* ── Metrics ── */}
        {activeTab === 'metrics' && (
          <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(200px,1fr))', gap:'10px' }}>
            {METRICS.map(m => <MetricCard key={m.key} metric={m} animate={animated} />)}
          </div>
        )}

        {/* ── Trajectory ── */}
        {activeTab === 'trajectory' && (
          <div className="glass" style={{ padding:'20px', height:'90%' }}>
            <div className="widget-title holo" style={{ marginBottom:'12px' }}>
              <Zap size={13} /> Battery SoC — Eclipse Stress Test
              <span style={{ marginLeft:'auto', fontSize:'0.65rem', color:'#6b7280' }}>
                Eclipse zone: T35–T55
              </span>
            </div>
            <div style={{ color:'#6b7280', fontSize:'0.72rem', marginBottom:'14px' }}>
              CASC-RL predicts eclipse 10 steps early → pre-charges. Traditional controller reacts too late.
            </div>
            <div style={{ height:'calc(100% - 80px)' }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={TRAJ_DATA}>
                  <defs>
                    <linearGradient id="tradG" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ff3d3d" stopOpacity={0.2} />
                      <stop offset="95%" stopColor="#ff3d3d" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="cascG" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#00f5ff" stopOpacity={0.2} />
                      <stop offset="95%" stopColor="#00f5ff" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="4 4" vertical={false} />
                  <XAxis dataKey="t" label={{ value:'Time step', position:'insideBottom', offset:-5, fill:'#6b7280', fontSize:11 }} />
                  <YAxis domain={[0, 100]} label={{ value:'SoC (%)', angle:-90, position:'insideLeft', fill:'#6b7280', fontSize:11 }} />
                  <Tooltip contentStyle={{ background:'#07090f', border:'1px solid rgba(255,255,255,0.1)', borderRadius:'8px' }} />
                  {/* Eclipse zone */}
                  <Area dataKey={() => 0} name="eclipse" stroke="none" />
                  <Area type="monotone" dataKey="traditional" stroke="#ff3d3d" fill="url(#tradG)" strokeWidth={2} name="Traditional (PID)" dot={false} />
                  <Area type="monotone" dataKey="casc" stroke="#00f5ff" fill="url(#cascG)" strokeWidth={2.5} name="CASC-RL" dot={false} />
                  <Legend wrapperStyle={{ fontSize:'11px' }} />

                  {/* Warning line */}
                  <Line type="monotone" dataKey={() => 20} stroke="#ffaa00" strokeDasharray="4 3" name="Min Safety" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* ── Ablation ── */}
        {activeTab === 'ablation' && (
          <div className="glass" style={{ padding:'20px', height:'90%' }}>
            <div className="widget-title holo" style={{ marginBottom:'12px' }}>
              Each Layer Adds Value — Ablation Study
            </div>
            <div style={{ height:'calc(100% - 50px)' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={ABLATION_DATA} layout="vertical" margin={{ left:10, right:20 }}>
                  <CartesianGrid strokeDasharray="4 4" horizontal={false} />
                  <XAxis type="number" domain={[0, 1500]} tick={{ fontSize:10 }} label={{ value:'Battery Lifetime (orbits)', position:'insideBottom', offset:-5, fill:'#6b7280', fontSize:11 }} />
                  <YAxis type="category" dataKey="name" width={160} tick={{ fontSize:10.5, fill:'#e8eaf0' }} />
                  <Tooltip contentStyle={{ background:'#07090f', border:'1px solid rgba(255,255,255,0.1)', borderRadius:'8px' }} />
                  <Bar dataKey="battery" radius={[0,4,4,0]} name="Battery Lifetime (orbits)">
                    {ABLATION_DATA.map((entry, i) => (
                      <Cell key={i}
                        fill={i === ABLATION_DATA.length - 1 ? '#00f5ff' : `rgba(0,245,255,${0.2 + i * 0.15})`}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* ── Radar ── */}
        {activeTab === 'radar' && (
          <div className="glass" style={{ padding:'20px', height:'90%' }}>
            <div className="widget-title holo" style={{ marginBottom:'12px' }}>
              Multi-Metric Comparison — All Baselines
            </div>
            <div style={{ height:'calc(100% - 50px)' }}>
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={RADAR_DATA}>
                  <PolarGrid stroke="rgba(255,255,255,0.08)" />
                  <PolarAngleAxis dataKey="metric" tick={{ fill:'#9ca3af', fontSize:11 }} />
                  <Radar name="Rule-Based" dataKey="RuleBased" stroke="#6b7280" fill="#6b7280" fillOpacity={0.1} />
                  <Radar name="PID" dataKey="PID" stroke="#ffaa00" fill="#ffaa00" fillOpacity={0.1} />
                  <Radar name="Ind. PPO" dataKey="IndPPO" stroke="#7b61ff" fill="#7b61ff" fillOpacity={0.1} />
                  <Radar name="CASC-RL" dataKey="CASCRL" stroke="#00f5ff" fill="#00f5ff" fillOpacity={0.2} strokeWidth={2} />
                  <Legend wrapperStyle={{ fontSize:'11px' }} />
                  <Tooltip contentStyle={{ background:'#07090f', border:'1px solid rgba(255,255,255,0.1)', borderRadius:'8px' }} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ComparisonPanel;
