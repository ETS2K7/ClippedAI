import Link from "next/link";

export default function NotFound() {
  return (
    <html lang="en">
      <body style={{ margin: 0, fontFamily: "system-ui, sans-serif", background: "#0a0a0a", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", minHeight: "100vh" }}>
        <div style={{ textAlign: "center" }}>
          <h1 style={{ fontSize: "6rem", fontWeight: 700, margin: 0, opacity: 0.15 }}>404</h1>
          <p style={{ fontSize: "1.25rem", margin: "1rem 0 2rem" }}>Page not found</p>
          <Link href="/dashboard" style={{ display: "inline-block", padding: "0.75rem 1.5rem", background: "#fff", color: "#000", borderRadius: "0.5rem", textDecoration: "none", fontWeight: 500 }}>
            Back to Dashboard
          </Link>
        </div>
      </body>
    </html>
  );
}
