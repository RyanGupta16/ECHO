"use client";

import { useSyncExternalStore } from "react";

/** Soft generative "tick" per word. Pitch encodes the model's confidence. Off by default. */
class Sound {
  enabled = false;
  private ctx: AudioContext | null = null;
  private listeners = new Set<() => void>();

  constructor() {
    if (typeof window === "undefined") return;
    try {
      this.enabled = localStorage.getItem("echo-sound") === "on";
    } catch {}
  }

  subscribe = (fn: () => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };

  toggle() {
    this.enabled = !this.enabled;
    try {
      localStorage.setItem("echo-sound", this.enabled ? "on" : "off");
    } catch {}
    if (this.enabled) this.ensure();
    this.listeners.forEach((f) => f());
  }

  private ensure() {
    this.ctx ??= new AudioContext();
    if (this.ctx.state === "suspended") this.ctx.resume();
    return this.ctx;
  }

  /** p = probability of the chosen word (0..1); voice offsets models slightly in pitch. */
  tick(p: number, voice = 0) {
    if (!this.enabled) return;
    const ctx = this.ensure();
    const now = ctx.currentTime + voice * 0.03; // lanes fire together → play as a soft arpeggio
    // Pentatonic so overlapping lanes stay consonant: confident words ring higher.
    const scale = [0, 2, 4, 7, 9, 12, 14, 16];
    const step = scale[Math.min(scale.length - 1, Math.floor(Math.sqrt(p) * scale.length))];
    const VOICES = [1, 1.5, 2, 1.25, 0.75]; // root, fifth, octave, third, low fifth — one per lane
    const freq = 330 * 2 ** (step / 12) * VOICES[voice % VOICES.length];
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    const filter = ctx.createBiquadFilter();
    osc.type = "triangle";
    osc.frequency.value = freq;
    filter.type = "lowpass";
    filter.frequency.value = 2400;
    gain.gain.setValueAtTime(0, now);
    gain.gain.linearRampToValueAtTime(0.045, now + 0.004);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.16);
    osc.connect(filter).connect(gain).connect(ctx.destination);
    osc.start(now);
    osc.stop(now + 0.18);
  }
}

let sound: Sound | null = null;
export function getSound() {
  sound ??= new Sound();
  return sound;
}

export function useSound() {
  const s = getSound();
  const enabled = useSyncExternalStore(s.subscribe, () => s.enabled, () => false);
  return { enabled, toggle: () => s.toggle(), tick: (p: number, v?: number) => s.tick(p, v) };
}
