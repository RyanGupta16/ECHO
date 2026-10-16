"use client";

import { useEffect, useState } from "react";

import { NAV, SITE } from "@/lib/content";
import { useSound } from "@/lib/sound";

export function Nav() {
  const [scrolled, setScrolled] = useState(false);
  const [active, setActive] = useState("");
  const sound = useSound();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    const io = new IntersectionObserver(
      (entries) =>
        entries.forEach((e) => {
          if (!e.isIntersecting) return;
          const href = "#" + e.target.id;
          setActive(NAV.some((n) => n.href === href) ? href : "");
        }),
      { rootMargin: "-45% 0px -50% 0px" },
    );
    document.querySelectorAll("main section[id], main header").forEach((el) => io.observe(el));
    return () => {
      window.removeEventListener("scroll", onScroll);
      io.disconnect();
    };
  }, []);

  return (
    <nav
      aria-label="Primary"
      className={`fixed inset-x-0 top-0 z-50 transition-all duration-300 ${scrolled ? "border-b border-line bg-bg/70 backdrop-blur-xl" : ""}`}
      style={{ paddingTop: "env(safe-area-inset-top, 0px)" }}
    >
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6">
        <a href="#top" className="flex items-center gap-2.5" aria-label="ECHO home">
          <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden>
            <circle cx="12" cy="12" r="9" fill="none" stroke="url(#g)" strokeWidth="2" />
            <circle cx="12" cy="12" r="4" fill="none" stroke="url(#g)" strokeWidth="1.5" opacity="0.6" />
            <defs>
              <linearGradient id="g" x1="0" x2="1">
                <stop offset="0" stopColor="#5ee7ff" />
                <stop offset="1" stopColor="#9b7bff" />
              </linearGradient>
            </defs>
          </svg>
          <span className="font-mono text-sm tracking-[0.3em]">{SITE.name}</span>
        </a>
        <ul className="hidden items-center gap-1 lg:flex">
          {NAV.map((n) => (
            <li key={n.href}>
              <a
                href={n.href}
                aria-current={active === n.href ? "true" : undefined}
                className={`rounded-full px-3.5 py-2 text-sm transition-colors ${active === n.href ? "bg-white/[0.06] text-ink" : "text-muted hover:text-ink"}`}
              >
                {n.label}
              </a>
            </li>
          ))}
        </ul>
        <div className="flex items-center gap-2">
        <button
          onClick={sound.toggle}
          aria-pressed={sound.enabled}
          aria-label={sound.enabled ? "Sound on — mute" : "Sound off — enable word sounds"}
          title="Word sounds"
          className={`grid h-10 w-10 place-items-center rounded-full border transition-colors ${sound.enabled ? "border-cyan/60 text-ink" : "border-line-2 text-ink-2 hover:text-ink"}`}
        >
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
            <path d="M2 6h2.5L8 3v10L4.5 10H2z" fill="currentColor" stroke="none" />
            {sound.enabled ? <path d="M10.5 5.5a3.5 3.5 0 010 5M12.5 3.5a6 6 0 010 9" /> : <path d="M11 6l4 4M15 6l-4 4" />}
          </svg>
        </button>
        <a
          href={SITE.repo}
          target="_blank"
          rel="noreferrer"
          className="inline-flex h-10 items-center gap-2 rounded-full border border-line-2 px-4 text-sm text-ink-2 transition-colors hover:border-ink-2 hover:text-ink"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden fill="currentColor">
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
          </svg>
          <span className="hidden sm:inline">Source</span>
        </a>
        </div>
      </div>
    </nav>
  );
}
