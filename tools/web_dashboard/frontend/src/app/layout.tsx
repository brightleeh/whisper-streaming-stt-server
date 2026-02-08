import { JetBrains_Mono, Space_Grotesk } from "next/font/google";
import "../styles/globals.css";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-sans",
  weight: ["400", "500", "600", "700"],
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  weight: ["400", "500", "600"],
});

export const metadata = {
  title: "Whisper Ops Dashboard",
  description: "Run gRPC load tests and watch metrics in real time.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${spaceGrotesk.variable} ${jetbrainsMono.variable}`}
      suppressHydrationWarning
    >
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  var key = "wops-theme";
                  var stored = localStorage.getItem(key);
                  var pref = stored === "light" || stored === "dark" || stored === "system" ? stored : "system";
                  var mql = window.matchMedia("(prefers-color-scheme: dark)");
                  var resolved = pref === "system" ? (mql.matches ? "dark" : "light") : pref;
                  document.documentElement.dataset.theme = resolved;
                } catch (e) {}
              })();
            `,
          }}
        />
      </head>
      <body style={{ fontFamily: "var(--font-sans)" }}>{children}</body>
    </html>
  );
}
