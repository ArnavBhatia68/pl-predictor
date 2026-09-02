import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PL Predictor — Gameweek Analytics",
  description: "Premier League outcome probabilities, expected goals, and an auditable live prediction record.",
  other: {
    "codex-preview": "pl-predictor",
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
