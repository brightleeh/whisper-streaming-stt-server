"use client";

import { useEffect, useState } from "react";

const THEME_KEY = "wops-theme";

type ThemePref = "system" | "light" | "dark";

function getSystemTheme() {
  if (typeof window === "undefined") return "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

function applyTheme(pref: ThemePref) {
  if (typeof document === "undefined") return;
  const resolved = pref === "system" ? getSystemTheme() : pref;
  document.documentElement.dataset.theme = resolved;
}

export default function ThemeToggle() {
  const [theme, setTheme] = useState<ThemePref>("system");

  useEffect(() => {
    let stored: ThemePref = "system";
    try {
      const saved = window.localStorage.getItem(THEME_KEY);
      if (saved === "light" || saved === "dark" || saved === "system") {
        stored = saved;
      }
    } catch {
      stored = "system";
    }
    setTheme(stored);
    applyTheme(stored);
  }, []);

  useEffect(() => {
    applyTheme(theme);
    try {
      window.localStorage.setItem(THEME_KEY, theme);
    } catch {
      // ignore
    }
  }, [theme]);

  useEffect(() => {
    if (theme !== "system") return;
    const mql = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => applyTheme("system");
    if (mql.addEventListener) {
      mql.addEventListener("change", handler);
      return () => mql.removeEventListener("change", handler);
    }
    mql.addListener(handler);
    return () => mql.removeListener(handler);
  }, [theme]);

  return (
    <div className="toggle-group" role="group" aria-label="Theme">
      <button
        type="button"
        className={`toggle-btn ${theme === "light" ? "active" : ""}`}
        onClick={() => setTheme("light")}
        aria-pressed={theme === "light"}
      >
        Light
      </button>
      <button
        type="button"
        className={`toggle-btn ${theme === "dark" ? "active" : ""}`}
        onClick={() => setTheme("dark")}
        aria-pressed={theme === "dark"}
      >
        Dark
      </button>
      <button
        type="button"
        className={`toggle-btn ${theme === "system" ? "active" : ""}`}
        onClick={() => setTheme("system")}
        aria-pressed={theme === "system"}
      >
        System
      </button>
    </div>
  );
}
