import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "katex/dist/katex.min.css";
import "./globals.css";
import { Providers } from "./providers";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "causal-ssm-agent",
    template: "%s | causal-ssm-agent",
  },
  description:
    "From research question to quantified treatment effects — powered by LLMs, state-space models, and Bayesian inference.",
  openGraph: {
    title: "causal-ssm-agent",
    description:
      "Pedagogical visualization of causal inference from question to treatment effects.",
    type: "website",
  },
  robots: {
    index: false,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <Providers>
          <main>{children}</main>
        </Providers>
      </body>
    </html>
  );
}
