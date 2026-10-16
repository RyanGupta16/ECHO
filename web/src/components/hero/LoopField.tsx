"use client";

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

const COUNT = 9000;

const vertex = /* glsl */ `
  uniform float uTime;
  uniform float uPulse;
  uniform vec2 uPointer;
  attribute float aAngle;
  attribute float aRadius;
  attribute float aSpeed;
  attribute float aSeed;
  varying float vMix;
  varying float vAlpha;

  void main() {
    float a = aAngle + uTime * aSpeed;
    // Recurrent loop: a ring with a slow breathing wobble
    float wobble = sin(a * 3.0 + uTime * 0.6 + aSeed * 6.28) * 0.12 + sin(a * 7.0 - uTime * 0.9) * 0.04;
    float r = aRadius + wobble;
    // Echo pulse: rings of particles thrown outward periodically
    float echo = fract(uTime * 0.12 + aSeed);
    float isEcho = step(0.86, aSeed);
    r += isEcho * echo * 2.6 * uPulse;
    vec3 p = vec3(cos(a) * r, sin(a) * r * 0.62, sin(a * 2.0 + aSeed * 3.0) * 0.35);
    p.xy += uPointer * 0.18 * (1.0 + aSeed);
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    float size = mix(1.4, 3.6, aSeed) * (1.0 - isEcho * echo * 0.6);
    gl_PointSize = size * (6.0 / -mv.z);
    vMix = 0.5 + 0.5 * sin(a + aSeed * 2.0);
    vAlpha = mix(0.75, 0.0, isEcho * echo) * (0.35 + 0.65 * aSeed);
  }
`;

const fragment = /* glsl */ `
  varying float vMix;
  varying float vAlpha;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float d = length(c);
    if (d > 0.5) discard;
    float glow = smoothstep(0.5, 0.0, d);
    vec3 cyan = vec3(0.37, 0.91, 1.0);
    vec3 violet = vec3(0.61, 0.48, 1.0);
    gl_FragColor = vec4(mix(cyan, violet, vMix), glow * vAlpha);
  }
`;

// Seeded PRNG (mulberry32): keeps the particle layout pure/deterministic across renders.
function mulberry32(a: number) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function Particles() {
  const material = useRef<THREE.ShaderMaterial>(null);
  const pointer = useRef(new THREE.Vector2());
  const { viewport } = useThree();

  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const angle = new Float32Array(COUNT);
    const radius = new Float32Array(COUNT);
    const speed = new Float32Array(COUNT);
    const seed = new Float32Array(COUNT);
    const rand = mulberry32(2026);
    for (let i = 0; i < COUNT; i++) {
      angle[i] = rand() * Math.PI * 2;
      const band = rand();
      radius[i] = 2.2 + (rand() - 0.5) * (band < 0.8 ? 0.35 : 1.4);
      speed[i] = 0.08 + rand() * 0.14;
      seed[i] = rand();
    }
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(COUNT * 3), 3));
    g.setAttribute("aAngle", new THREE.BufferAttribute(angle, 1));
    g.setAttribute("aRadius", new THREE.BufferAttribute(radius, 1));
    g.setAttribute("aSpeed", new THREE.BufferAttribute(speed, 1));
    g.setAttribute("aSeed", new THREE.BufferAttribute(seed, 1));
    return g;
  }, []);

  const uniforms = useMemo(
    () => ({ uTime: { value: 0 }, uPulse: { value: 1 }, uPointer: { value: new THREE.Vector2() } }),
    [],
  );

  useFrame((state, delta) => {
    if (!material.current) return;
    material.current.uniforms.uTime.value += delta;
    pointer.current.lerp(state.pointer, 0.04);
    material.current.uniforms.uPointer.value.copy(pointer.current);
  });

  return (
    <points geometry={geometry} scale={Math.min(1, viewport.width / 7)} frustumCulled={false}>
      <shaderMaterial
        ref={material}
        vertexShader={vertex}
        fragmentShader={fragment}
        uniforms={uniforms}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

export default function LoopField() {
  return (
    <Canvas
      camera={{ position: [0, 0, 6], fov: 50 }}
      dpr={[1, 1.75]}
      gl={{ antialias: false, alpha: true, powerPreference: "high-performance" }}
      aria-hidden
    >
      <Particles />
    </Canvas>
  );
}
