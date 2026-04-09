import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import fs from "fs";
import path from "path";

// Map font names to public CDN sources for fonts not bundled locally.
// For locally bundled fonts, we serve from public/fonts/.
const FONT_CDN_MAP: Record<string, string> = {
  "TikTokSans-Regular":    "https://cdn.jsdelivr.net/fontsource/fonts/noto-sans@latest/latin-400-normal.woff2",
  "TikTokSans-Bold":       "https://cdn.jsdelivr.net/fontsource/fonts/noto-sans@latest/latin-700-normal.woff2",
  "TikTokSans-SemiBold":   "https://cdn.jsdelivr.net/fontsource/fonts/noto-sans@latest/latin-600-normal.woff2",
  "TikTokSans-Medium":     "https://cdn.jsdelivr.net/fontsource/fonts/noto-sans@latest/latin-500-normal.woff2",
  "Montserrat-Regular":    "https://cdn.jsdelivr.net/fontsource/fonts/montserrat@latest/latin-400-normal.woff2",
  "Montserrat-Bold":       "https://cdn.jsdelivr.net/fontsource/fonts/montserrat@latest/latin-700-normal.woff2",
  "Montserrat-SemiBold":   "https://cdn.jsdelivr.net/fontsource/fonts/montserrat@latest/latin-600-normal.woff2",
  "Montserrat-BoldItalic": "https://cdn.jsdelivr.net/fontsource/fonts/montserrat@latest/latin-700-italic.woff2",
  "Roboto-Regular":        "https://cdn.jsdelivr.net/fontsource/fonts/roboto@latest/latin-400-normal.woff2",
  "Roboto-Bold":           "https://cdn.jsdelivr.net/fontsource/fonts/roboto@latest/latin-700-normal.woff2",
  "Roboto-Medium":         "https://cdn.jsdelivr.net/fontsource/fonts/roboto@latest/latin-500-normal.woff2",
  "OpenSans-Regular":      "https://cdn.jsdelivr.net/fontsource/fonts/open-sans@latest/latin-400-normal.woff2",
  "OpenSans-Bold":         "https://cdn.jsdelivr.net/fontsource/fonts/open-sans@latest/latin-700-normal.woff2",
  "OpenSans-SemiBold":     "https://cdn.jsdelivr.net/fontsource/fonts/open-sans@latest/latin-600-normal.woff2",
  "Lato-Regular":          "https://cdn.jsdelivr.net/fontsource/fonts/lato@latest/latin-400-normal.woff2",
  "Lato-Bold":             "https://cdn.jsdelivr.net/fontsource/fonts/lato@latest/latin-700-normal.woff2",
  "Inter-Regular":         "https://cdn.jsdelivr.net/fontsource/fonts/inter@latest/latin-400-normal.woff2",
  "Inter-Bold":            "https://cdn.jsdelivr.net/fontsource/fonts/inter@latest/latin-700-normal.woff2",
  "Inter-SemiBold":        "https://cdn.jsdelivr.net/fontsource/fonts/inter@latest/latin-600-normal.woff2",
  "Raleway-Regular":       "https://cdn.jsdelivr.net/fontsource/fonts/raleway@latest/latin-400-normal.woff2",
  "Raleway-Bold":          "https://cdn.jsdelivr.net/fontsource/fonts/raleway@latest/latin-700-normal.woff2",
};

// Set of allowed font names — prevents path traversal via filesystem access
const ALLOWED_FONT_NAMES = new Set(Object.keys(FONT_CDN_MAP));

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ name: string }> }
) {
  const session = await auth();
  if (!session?.user?.id) {
    return new NextResponse(null, { status: 401 });
  }

  const { name } = await params;
  const fontName = name.replace(/\.(ttf|otf|woff2?)$/i, "");

  // Path traversal guard: only serve fonts from the known allowlist
  if (!ALLOWED_FONT_NAMES.has(fontName)) {
    return new NextResponse(null, { status: 404 });
  }

  // 1. Try to serve from local public/fonts/ first
  const fontsDir = path.resolve(process.cwd(), "public", "fonts");
  const localPath = path.resolve(fontsDir, `${fontName}.ttf`);

  // Defense-in-depth: ensure resolved path is still within the fonts directory
  if (!localPath.startsWith(fontsDir + path.sep)) {
    return new NextResponse(null, { status: 400 });
  }

  if (fs.existsSync(localPath)) {
    const fileBuffer = fs.readFileSync(localPath);
    return new NextResponse(fileBuffer, {
      status: 200,
      headers: {
        "Content-Type": "font/ttf",
        "Cache-Control": "public, max-age=31536000, immutable",
      },
    });
  }

  // 2. Proxy from CDN map
  const cdnUrl = FONT_CDN_MAP[fontName];
  if (cdnUrl) {
    try {
      const response = await fetch(cdnUrl, {
        headers: { "User-Agent": "Mozilla/5.0" },
      });
      if (response.ok) {
        const arrayBuffer = await response.arrayBuffer();
        const contentType = cdnUrl.includes(".woff2") ? "font/woff2" : "font/ttf";
        return new NextResponse(arrayBuffer, {
          status: 200,
          headers: {
            "Content-Type": contentType,
            "Cache-Control": "public, max-age=31536000, immutable",
          },
        });
      }
    } catch {
      // fall through to 404
    }
  }

  return new NextResponse(null, { status: 404 });
}
