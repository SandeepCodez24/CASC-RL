
import React, { useRef, useMemo, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Stars, OrbitControls, PerspectiveCamera, useTexture, Html, Line } from '@react-three/drei';
import * as THREE from 'three';
import { propagateOrbit, isSatelliteInEclipse, R_EARTH } from '../../engine/physics.js';

const SCALE = 1 / 1.4e6;
const EARTH_R = R_EARTH * SCALE;

const MODE_COLORS = {
  PAYLOAD:     '#00ff88',
  RELAY:       '#7b61ff',
  CHARGE:      '#f5d800',
  HIBERNATE:   '#ff3d3d',
  PAYLOAD_OFF: '#6b7280',
};

// ─── Individual Satellite ────────────────────────────────────────────────────
const SatelliteNode = ({ sat, simTime, isSelected, onSelect }) => {
  const meshRef = useRef();
  const [hovered, setHovered] = useState(false);
  const color = sat.status === 'CRITICAL' ? '#ff3d3d' : (MODE_COLORS[sat.mode] || '#00f5ff');

  const position = useMemo(() => {
    const p = propagateOrbit(sat.elements, simTime);
    return [p.x * SCALE, p.z * SCALE, p.y * SCALE];
  }, [sat.elements, simTime]);

  const inEclipse = useMemo(() => {
    const p = propagateOrbit(sat.elements, simTime);
    return isSatelliteInEclipse(p);
  }, [sat.elements, simTime]);

  useFrame((state) => {
    if (meshRef.current) {
      const pulse = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.15;
      meshRef.current.scale.setScalar(isSelected ? pulse * 1.3 : hovered ? 1.2 : 1);
    }
  });

  return (
    <group position={position}>
      {/* Satellite body */}
      <mesh
        ref={meshRef}
        onClick={() => onSelect(sat.id)}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <boxGeometry args={[0.05, 0.028, 0.028]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isSelected ? 1.5 : 0.7}
          metalness={0.8}
          roughness={0.2}
        />
      </mesh>

      {/* Solar panels */}
      <mesh position={[0.09, 0, 0]}>
        <boxGeometry args={[0.07, 0.002, 0.045]} />
        <meshStandardMaterial
          color={inEclipse ? '#112233' : '#1a44cc'}
          emissive={inEclipse ? '#000' : '#1133cc'}
          emissiveIntensity={0.4}
        />
      </mesh>
      <mesh position={[-0.09, 0, 0]}>
        <boxGeometry args={[0.07, 0.002, 0.045]} />
        <meshStandardMaterial
          color={inEclipse ? '#112233' : '#1a44cc'}
          emissive={inEclipse ? '#000' : '#1133cc'}
          emissiveIntensity={0.4}
        />
      </mesh>

      {/* Selection ring */}
      {isSelected && (
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[0.11, 0.006, 8, 32]} />
          <meshBasicMaterial color={color} transparent opacity={0.8} />
        </mesh>
      )}

      {/* Point light */}
      <pointLight color={color} intensity={isSelected ? 0.7 : 0.3} distance={1.0} />

      {/* Hover tooltip */}
      {(hovered || isSelected) && (
        <Html
          center
          distanceFactor={9}
          style={{ pointerEvents: 'none', width: '170px' }}
        >
          <div style={{
            background: 'rgba(4,6,12,0.96)',
            border: `1px solid ${color}44`,
            borderRadius: '8px',
            padding: '10px 12px',
            fontSize: '11px',
            fontFamily: "'JetBrains Mono', monospace",
            transform: 'translateY(-80px)',
            boxShadow: `0 4px 20px rgba(0,0,0,0.9), 0 0 12px ${color}22`,
            backdropFilter: 'blur(8px)',
          }}>
            <div style={{ color, fontWeight: 700, marginBottom: '7px', borderBottom: `1px solid ${color}33`, paddingBottom: '5px', fontSize: '12px' }}>
              {sat.id}
            </div>
            {[
              { k: 'SoC',  v: `${sat.soc.toFixed(1)}%`,       c: sat.soc < 15 ? '#ff3d3d' : '#00f5ff' },
              { k: 'SoH',  v: `${((sat.soh||1)*100).toFixed(1)}%`, c: '#00ff88' },
              { k: 'Temp', v: `${sat.temp.toFixed(1)}°C`,     c: sat.temp > 55 ? '#ff3d3d' : sat.temp > 40 ? '#ffaa00' : '#e8eaf0' },
              { k: 'Mode', v: sat.mode,                         c: color },
            ].map(({ k, v, c }) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px', color: '#e8eaf0' }}>
                <span style={{ color: '#6b7280' }}>{k}</span>
                <span style={{ color: c, fontWeight: 600 }}>{v}</span>
              </div>
            ))}
            {inEclipse && (
              <div style={{ marginTop: '6px', color: '#ffaa00', fontSize: '10px', borderTop: `1px solid ${color}22`, paddingTop: '4px' }}>
                🌑 IN ECLIPSE
              </div>
            )}
          </div>
        </Html>
      )}
    </group>
  );
};

// ─── ISL Links ───────────────────────────────────────────────────────────────
const ISLLinks = ({ sats, simTime }) => {
  const links = useMemo(() => {
    const result = [];
    for (let i = 0; i < sats.length; i++) {
      const j = (i + 1) % sats.length;
      const p1 = propagateOrbit(sats[i].elements, simTime);
      const p2 = propagateOrbit(sats[j].elements, simTime);
      const dist = Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2);
      const active = dist < 4000e3;
      const relay = sats[i].mode === 'RELAY' || sats[j].mode === 'RELAY';
      result.push({
        from: [p1.x * SCALE, p1.z * SCALE, p1.y * SCALE],
        to:   [p2.x * SCALE, p2.z * SCALE, p2.y * SCALE],
        active, relay,
      });
    }
    return result;
  }, [sats, simTime]);

  return (
    <>
      {links.map((link, i) => (
        <Line
          key={i}
          points={[link.from, link.to]}
          color={link.relay ? '#7b61ff' : link.active ? '#00f5ff' : '#1a2030'}
          lineWidth={link.relay ? 2.5 : 1}
          transparent
          opacity={link.relay ? 0.75 : link.active ? 0.35 : 0.08}
        />
      ))}
    </>
  );
};

// ─── Orbit Path ───────────────────────────────────────────────────────────────
const OrbitPath = ({ elements, color }) => {
  const points = useMemo(() => {
    const pts = [];
    for (let t = 0; t <= 5700; t += 90) {
      const p = propagateOrbit(elements, t);
      pts.push(new THREE.Vector3(p.x * SCALE, p.z * SCALE, p.y * SCALE));
    }
    // Close the loop
    const p0 = propagateOrbit(elements, 0);
    pts.push(new THREE.Vector3(p0.x * SCALE, p0.z * SCALE, p0.y * SCALE));
    return pts;
  }, [elements]);

  return (
    <Line
      points={points}
      color={color}
      lineWidth={0.4}
      transparent
      opacity={0.10}
    />
  );
};

// ─── Earth (no try/catch around hooks!) ──────────────────────────────────────
const EarthTextureMesh = () => {
  // useTexture must NOT be inside try/catch — it uses React Suspense
  const earthTex = useTexture('/earth.png');
  const earthRef = useRef();

  useFrame((_, delta) => {
    if (earthRef.current) earthRef.current.rotation.y += delta * 0.025;
  });

  return (
    <mesh ref={earthRef} rotation={[0, -Math.PI / 2, 0.15]}>
      <sphereGeometry args={[EARTH_R, 64, 64]} />
      <meshStandardMaterial
        map={earthTex}
        emissive="#0a1a30"
        emissiveIntensity={0.15}
        roughness={0.75}
        metalness={0.1}
      />
    </mesh>
  );
};

const EarthFallbackMesh = () => {
  const earthRef = useRef();
  useFrame((_, delta) => {
    if (earthRef.current) earthRef.current.rotation.y += delta * 0.025;
  });
  return (
    <mesh ref={earthRef}>
      <sphereGeometry args={[EARTH_R, 48, 48]} />
      <meshStandardMaterial color="#1a4a8a" emissive="#0a2040" emissiveIntensity={0.3} roughness={0.6} />
    </mesh>
  );
};

// Wrapper that uses React.Suspense to safely load texture
const Earth = () => {
  return (
    <group>
      {/* Atmosphere glow */}
      <mesh>
        <sphereGeometry args={[EARTH_R * 1.08, 48, 48]} />
        <meshBasicMaterial color="#0033cc" transparent opacity={0.04} side={THREE.BackSide} />
      </mesh>

      {/* Earth with suspense fallback */}
      <React.Suspense fallback={<EarthFallbackMesh />}>
        <EarthTextureMesh />
      </React.Suspense>

      {/* Grid overlay */}
      <mesh>
        <sphereGeometry args={[EARTH_R * 1.006, 24, 24]} />
        <meshBasicMaterial color="#00f5ff" wireframe transparent opacity={0.04} />
      </mesh>

      {/* Eclipse shadow cone (simplified umbra along -X) */}
      <mesh position={[-EARTH_R * 3.2, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
        <coneGeometry args={[EARTH_R * 0.99, EARTH_R * 7, 32, 1, true]} />
        <meshBasicMaterial color="#00000f" transparent opacity={0.4} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
};

// ─── Main CelestialView ───────────────────────────────────────────────────────
const CelestialView = ({ sats, simTime, selectedSatId, onSelect }) => {
  return (
    <div style={{ width: '100%', height: '100%', background: '#04060c' }}>
      <Canvas
        gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
        dpr={[1, 1.5]}
        onCreated={({ gl }) => {
          gl.setClearColor('#04060c', 1);
        }}
      >
        <PerspectiveCamera makeDefault position={[0, EARTH_R * 3.5, EARTH_R * 4.5]} fov={48} near={0.001} far={500} />
        <color attach="background" args={['#04060c']} />

        {/* Lighting: Sun from +X */}
        <ambientLight intensity={0.25} />
        <directionalLight position={[12, 2, 4]} intensity={2.2} color="#fff9f0" />

        <Earth />

        {/* Orbit paths for first 4 sats (representative planes) */}
        {sats.slice(0, 4).map(sat => (
          <OrbitPath key={sat.id} elements={sat.elements} color={sat.color} />
        ))}

        {/* ISL Links */}
        <ISLLinks sats={sats} simTime={simTime} />

        {/* All Satellites */}
        {sats.map(sat => (
          <SatelliteNode
            key={sat.id}
            sat={sat}
            simTime={simTime}
            isSelected={sat.id === selectedSatId}
            onSelect={onSelect || (() => {})}
          />
        ))}

        <Stars radius={200} depth={80} count={7000} factor={5} saturation={0} fade speed={0.4} />

        <OrbitControls
          enableDamping
          dampingFactor={0.07}
          minDistance={EARTH_R * 1.6}
          maxDistance={EARTH_R * 14}
        />
      </Canvas>
    </div>
  );
};

export default CelestialView;
