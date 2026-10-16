"use client";

/** Renders a 1200×630 social card of an Arena story and shares or downloads it. */

type Card = { model: string; color: string; seed: string; text: string; temperature: number; url: string };

function cssVar(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function resolveColor(c: string) {
  return c.startsWith("var(") ? cssVar(c.slice(4, -1)) : c;
}

function wrap(ctx: CanvasRenderingContext2D, text: string, maxW: number) {
  const words = text.split(" ");
  const lines: string[] = [];
  let line = "";
  for (const w of words) {
    const t = line ? `${line} ${w}` : w;
    if (ctx.measureText(t).width > maxW && line) {
      lines.push(line);
      line = w;
    } else line = t;
  }
  if (line) lines.push(line);
  return lines;
}

export async function renderCard(c: Card): Promise<Blob> {
  const W = 1200, H = 630;
  const canvas = document.createElement("canvas");
  canvas.width = W * 2;
  canvas.height = H * 2;
  const ctx = canvas.getContext("2d")!;
  ctx.scale(2, 2);
  await document.fonts.ready;
  const serif = cssVar("--font-instrument-serif") || "Georgia, serif";
  const sans = cssVar("--font-geist-sans") || "system-ui, sans-serif";
  const mono = cssVar("--font-geist-mono") || "ui-monospace, monospace";
  const accent = resolveColor(c.color);

  // Background + glow + echo rings
  ctx.fillStyle = "#05060a";
  ctx.fillRect(0, 0, W, H);
  const g = ctx.createRadialGradient(W * 0.86, H * 0.12, 0, W * 0.86, H * 0.12, 520);
  g.addColorStop(0, accent + "55");
  g.addColorStop(1, "transparent");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);
  for (let i = 0; i < 5; i++) {
    ctx.strokeStyle = `rgba(155,123,255,${0.16 - i * 0.03})`;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.ellipse(W * 0.86, H * 0.12, 90 + i * 70, 56 + i * 44, 0, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Header
  ctx.fillStyle = "#f2f1ee";
  ctx.font = `500 22px ${mono}`;
  ctx.fillText("E C H O", 64, 82);
  ctx.fillStyle = "#7f8499";
  ctx.font = `16px ${mono}`;
  ctx.fillText("RECURRENT TEXT LAB", 168, 82);

  // Model chip
  ctx.font = `500 20px ${sans}`;
  const label = `${c.model}  ·  temperature ${c.temperature.toFixed(2)}`;
  const lw = ctx.measureText(label).width;
  ctx.fillStyle = "rgba(255,255,255,0.06)";
  ctx.beginPath();
  ctx.roundRect(64, 116, lw + 58, 44, 22);
  ctx.fill();
  ctx.fillStyle = accent;
  ctx.beginPath();
  ctx.arc(88, 138, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#f2f1ee";
  ctx.fillText(label, 104, 145);

  // Story text: seed muted, continuation bright — shrink to fit.
  let size = 40;
  let lines: { t: string; seed: boolean }[] = [];
  for (; size >= 22; size -= 2) {
    ctx.font = `${size}px ${serif}`;
    const all = wrap(ctx, `${c.seed} ${c.text}`, W - 128);
    const seedLen = c.seed.length;
    let count = 0;
    lines = all.map((t) => {
      const seed = count < seedLen;
      count += t.length + 1;
      return { t, seed };
    });
    if (lines.length * size * 1.32 < H - 300) break;
  }
  let y = 230;
  ctx.font = `${size}px ${serif}`;
  let consumed = 0;
  for (const { t } of lines) {
    // colour seed part of each line muted, the rest bright
    const seedChars = Math.max(0, Math.min(t.length, c.seed.length - consumed));
    const a = t.slice(0, seedChars);
    const b = t.slice(seedChars);
    ctx.fillStyle = "#7f8499";
    ctx.fillText(a, 64, y);
    ctx.fillStyle = "#f2f1ee";
    ctx.fillText(b, 64 + ctx.measureText(a).width, y);
    consumed += t.length + 1;
    y += size * 1.32;
  }

  // Footer
  ctx.strokeStyle = "#1a1f2e";
  ctx.beginPath();
  ctx.moveTo(64, H - 84);
  ctx.lineTo(W - 64, H - 84);
  ctx.stroke();
  ctx.fillStyle = "#b9bccb";
  ctx.font = `18px ${sans}`;
  ctx.fillText("Written live in the browser by a neural network trained on 30,000 children's stories.", 64, H - 46);
  ctx.fillStyle = "#5ee7ff";
  ctx.font = `16px ${mono}`;
  const u = c.url.replace(/^https?:\/\//, "");
  ctx.fillText(u, W - 64 - ctx.measureText(u).width, H - 46);

  return new Promise((res) => canvas.toBlob((b) => res(b!), "image/png"));
}

export async function shareCard(c: Card) {
  const blob = await renderCard(c);
  const file = new File([blob], `echo-${c.model.toLowerCase().replace(/\s+/g, "-")}.png`, { type: "image/png" });
  if (navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ files: [file], title: "ECHO — a story by a neural network", url: c.url });
      return "shared";
    } catch {
      /* user cancelled — fall through to download */
    }
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = file.name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  return "downloaded";
}
