import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Live Fact Checker",
  description: "Real-time claim detection and verification dashboard",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}

