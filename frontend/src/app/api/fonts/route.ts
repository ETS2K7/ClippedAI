import { NextResponse } from "next/server";
import { auth } from "~/server/auth";

// ClippedAI built-in font list matching the Python backend defaults
const BUILT_IN_FONTS = [
  { name: "TikTokSans-Regular", display_name: "TikTok Sans Regular", format: "ttf" },
  { name: "TikTokSans-Bold", display_name: "TikTok Sans Bold", format: "ttf" },
  { name: "TikTokSans-SemiBold", display_name: "TikTok Sans SemiBold", format: "ttf" },
  { name: "TikTokSans-Medium", display_name: "TikTok Sans Medium", format: "ttf" },
  { name: "Montserrat-Regular", display_name: "Montserrat Regular", format: "ttf" },
  { name: "Montserrat-Bold", display_name: "Montserrat Bold", format: "ttf" },
  { name: "Montserrat-SemiBold", display_name: "Montserrat SemiBold", format: "ttf" },
  { name: "Montserrat-BoldItalic", display_name: "Montserrat Bold Italic", format: "ttf" },
  { name: "Roboto-Regular", display_name: "Roboto Regular", format: "ttf" },
  { name: "Roboto-Bold", display_name: "Roboto Bold", format: "ttf" },
  { name: "Roboto-Medium", display_name: "Roboto Medium", format: "ttf" },
  { name: "OpenSans-Regular", display_name: "Open Sans Regular", format: "ttf" },
  { name: "OpenSans-Bold", display_name: "Open Sans Bold", format: "ttf" },
  { name: "OpenSans-SemiBold", display_name: "Open Sans SemiBold", format: "ttf" },
  { name: "Lato-Regular", display_name: "Lato Regular", format: "ttf" },
  { name: "Lato-Bold", display_name: "Lato Bold", format: "ttf" },
  { name: "Inter-Regular", display_name: "Inter Regular", format: "ttf" },
  { name: "Inter-Bold", display_name: "Inter Bold", format: "ttf" },
  { name: "Inter-SemiBold", display_name: "Inter SemiBold", format: "ttf" },
  { name: "Raleway-Regular", display_name: "Raleway Regular", format: "ttf" },
  { name: "Raleway-Bold", display_name: "Raleway Bold", format: "ttf" },
];

export async function GET() {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  return NextResponse.json({ fonts: BUILT_IN_FONTS });
}
