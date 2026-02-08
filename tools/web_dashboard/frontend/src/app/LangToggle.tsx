"use client";

import type { Locale } from "./i18n";

const OPTIONS: { value: Locale; label: string }[] = [
  { value: "en", label: "EN" },
  { value: "ko", label: "KO" },
  { value: "ja", label: "JA" },
];

export default function LangToggle({
  value,
  onChange,
}: {
  value: Locale;
  onChange: (next: Locale) => void;
}) {
  return (
    <div className="toggle-group" role="group" aria-label="Language">
      {OPTIONS.map((option) => (
        <button
          key={option.value}
          type="button"
          className={`toggle-btn ${value === option.value ? "active" : ""}`}
          onClick={() => onChange(option.value)}
          aria-pressed={value === option.value}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
