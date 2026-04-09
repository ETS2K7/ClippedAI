import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import fs from "fs";
import path from "path";

// Map font names to public CDN sources for fonts not bundled locally.
// For locally bundled fonts, we serve from public/fonts/.
const FONT_CDN_MAP: Record<string, string> = {
  "TikTokSans-Regular":    "https://fonts.gstatic.com/s/notosans/v36/o-0NIpQlx3QUlC5A4PNjThZVZNyB.woff2",
  "TikTokSans-Bold":       "https://fonts.gstatic.com/s/notosans/v36/o-0TIpQlx3QUlC5A4PNjFhZVZNyB.woff2",
  "TikTokSans-SemiBold":   "https://fonts.gstatic.com/s/notosans/v36/o-0TIpQlx3QUlC5A4PNjFhZVZNyB.woff2",
  "TikTokSans-Medium":     "https://fonts.gstatic.com/s/notosans/v36/o-0TIpQlx3QUlC5A4PNjFhZVZNyB.woff2",
  "Montserrat-Regular":    "https://fonts.gstatic.com/s/montserrat/v26/JTUSjIg1_i6t8kCHKm459WlhyyTh89Y.woff2",
  "Montserrat-Bold":       "https://fonts.gstatic.com/s/montserrat/v26/JTURjIg1_i6t8kCHKm45_dJE3gnD_g.woff2",
  "Montserrat-SemiBold":   "https://fonts.gstatic.com/s/montserrat/v26/JTURjIg1_i6t8kCHKm45_bZF3gnD_g.woff2",
  "Montserrat-BoldItalic": "https://fonts.gstatic.com/s/montserrat/v26/JTURjIg1_i6t8kCHKm45_dJE7gnD_g.woff2",
  "Roboto-Regular":        "https://fonts.gstatic.com/s/roboto/v30/KFOmCnqEu92Fr1Mu4mxKKTU1Kg.woff2",
  "Roboto-Bold":           "https://fonts.gstatic.com/s/roboto/v30/KFOlCnqEu92Fr1MmWUlfBBc4AMP6lQ.woff2",
  "Roboto-Medium":         "https://fonts.gstatic.com/s/roboto/v30/KFOlCnqEu92Fr1MmEU9fBBc4AMP6lQ.woff2",
  "OpenSans-Regular":      "https://fonts.gstatic.com/s/opensans/v40/memSYaGs126MiZpBA-UvWbX2vVnXBbObj2OVZyOOSr4dVJWUgsiH0B4gaVI.woff2",
  "OpenSans-Bold":         "https://fonts.gstatic.com/s/opensans/v40/memSYaGs126MiZpBA-UvWbX2vVnXBbObj2OVZyOOSr4dVJWUgsg-1x4gaVI.woff2",
  "OpenSans-SemiBold":     "https://fonts.gstatic.com/s/opensans/v40/memSYaGs126MiZpBA-UvWbX2vVnXBbObj2OVZyOOSr4dVJWUgsm-Bx4gaVI.woff2",
  "Lato-Regular":          "https://fonts.gstatic.com/s/lato/v24/S6uyw4BMUTPHjx4wXiWtFCc.woff2",
  "Lato-Bold":             "https://fonts.gstatic.com/s/lato/v24/S6u9w4BMUTPHh6UVSwiPGQ3q5d0.woff2",
  "Inter-Regular":         "https://fonts.gstatic.com/s/inter/v13/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuLyfAZ9hiA.woff2",
  "Inter-Bold":            "https://fonts.gstatic.com/s/inter/v13/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuFuYAZ9hiA.woff2",
  "Inter-SemiBold":        "https://fonts.gstatic.com/s/inter/v13/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuGKYAZ9hiA.woff2",
  "Raleway-Regular":       "https://fonts.gstatic.com/s/raleway/v28/1Ptxg8zYS_SKggPN4iEgvnHyvveLxVvaorCFPrEHJA.woff2",
  "Raleway-Bold":          "https://fonts.gstatic.com/s/raleway/v28/1Ptxg8zYS_SKggPN4iEgvnHyvveLxVvaorCIPbEHJA.woff2",
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
