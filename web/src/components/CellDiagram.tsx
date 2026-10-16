import type { Kind } from "@/lib/engine/protocol";

/** Small animated schematic of each recurrent cell. Pure SVG + CSS. */
export function CellDiagram({ kind, color }: { kind: Kind; color: string }) {
  const common = { stroke: "var(--line-2)", strokeWidth: 1.5, fill: "none" } as const;
  const pulse = { stroke: color, strokeWidth: 2, fill: "none", strokeLinecap: "round" as const };

  return (
    <svg viewBox="0 0 240 120" className="h-full w-full" role="img" aria-label={`${kind} cell diagram`}>
      <style>{`
        .flow{stroke-dasharray:10 190;animation:flow 2.4s linear infinite}
        .flow.rev{animation-direction:reverse}
        .gate{transform-box:fill-box;transform-origin:center;animation:gate 3.2s ease-in-out infinite}
        @keyframes flow{to{stroke-dashoffset:-200}}
        @keyframes gate{0%,100%{transform:scaleY(.25)}50%{transform:scaleY(1)}}
        @media (prefers-reduced-motion:reduce){.flow,.gate{animation:none}}
      `}</style>

      {kind === "rnn" && (
        <g>
          <path d="M20 60 H95" {...common} />
          <path d="M145 60 H220" {...common} />
          <rect x="95" y="38" width="50" height="44" rx="10" {...common} />
          <text x="120" y="65" textAnchor="middle" fontSize="12" fill="var(--ink-2)" fontFamily="var(--font-geist-mono)">tanh</text>
          <path d="M145 50 C185 10, 55 10, 95 50" {...common} />
          <path d="M145 50 C185 10, 55 10, 95 50" {...pulse} className="flow" />
          <path d="M20 60 H95" {...pulse} className="flow" />
          <text x="20" y="80" fontSize="10" fill="var(--muted)" fontFamily="var(--font-geist-mono)">xₜ</text>
          <text x="205" y="80" fontSize="10" fill="var(--muted)" fontFamily="var(--font-geist-mono)">hₜ</text>
        </g>
      )}

      {kind === "birnn" && (
        <g>
          {[30, 75, 120, 165, 210].map((x) => (
            <g key={x}>
              <circle cx={x} cy="40" r="9" {...common} />
              <circle cx={x} cy="80" r="9" {...common} />
            </g>
          ))}
          <path d="M39 40 H201" {...common} />
          <path d="M39 80 H201" {...common} />
          <path d="M39 40 H201" {...pulse} className="flow" />
          <path d="M39 80 H201" {...pulse} className="flow rev" strokeOpacity={0.7} />
          <text x="30" y="20" fontSize="10" fill="var(--muted)" fontFamily="var(--font-geist-mono)">forward →</text>
          <text x="150" y="108" fontSize="10" fill="var(--muted)" fontFamily="var(--font-geist-mono)">← backward</text>
        </g>
      )}

      {kind === "lstm" && (
        <g>
          <path d="M10 28 H230" {...common} strokeWidth={2.5} />
          <path d="M10 28 H230" {...pulse} className="flow" strokeWidth={3} />
          <text x="12" y="18" fontSize="10" fill="var(--muted)" fontFamily="var(--font-geist-mono)">cell state cₜ — the conveyor belt</text>
          {[
            { x: 60, l: "forget", d: "0s" },
            { x: 120, l: "input", d: "-1s" },
            { x: 180, l: "output", d: "-2s" },
          ].map((g) => (
            <g key={g.l}>
              <path d={`M${g.x} 28 V62`} {...common} />
              <rect x={g.x - 14} y="62" width="28" height="30" rx="6" {...common} />
              <rect x={g.x - 9} y="67" width="18" height="20" rx="3" fill={color} opacity={0.75} className="gate" style={{ animationDelay: g.d }} />
              <text x={g.x} y="108" textAnchor="middle" fontSize="10" fill="var(--ink-2)" fontFamily="var(--font-geist-mono)">{g.l}</text>
            </g>
          ))}
        </g>
      )}

      {kind === "gru" && (
        <g>
          <path d="M10 40 H230" {...common} strokeWidth={2} />
          <path d="M10 40 H230" {...pulse} className="flow" />
          <text x="12" y="28" fontSize="10" fill="var(--muted)" fontFamily="var(--font-geist-mono)">hidden state hₜ</text>
          {[
            { x: 85, l: "reset", d: "0s" },
            { x: 155, l: "update", d: "-1.6s" },
          ].map((g) => (
            <g key={g.l}>
              <path d={`M${g.x} 40 V62`} {...common} />
              <rect x={g.x - 14} y="62" width="28" height="30" rx="6" {...common} />
              <rect x={g.x - 9} y="67" width="18" height="20" rx="3" fill={color} opacity={0.75} className="gate" style={{ animationDelay: g.d }} />
              <text x={g.x} y="108" textAnchor="middle" fontSize="10" fill="var(--ink-2)" fontFamily="var(--font-geist-mono)">{g.l}</text>
            </g>
          ))}
        </g>
      )}
      {kind === "tfm" && (
        <g>
          {[30, 75, 120, 165, 210].map((x) => (
            <circle key={x} cx={x} cy="88" r="8" {...common} />
          ))}
          {[30, 75, 120, 165].map((x, i) => (
            <path
              key={x}
              d={`M${x} 80 Q${(x + 210) / 2} ${20 + i * 10} 210 80`}
              {...pulse}
              strokeOpacity={0.35 + i * 0.18}
              className="flow"
              style={{ animationDelay: `${-i * 0.5}s` }}
            />
          ))}
          <text x="12" y="20" fontSize="10" fill="var(--muted)" fontFamily="var(--font-geist-mono)">every word attends to every earlier word</text>
        </g>
      )}
    </svg>
  );
}
