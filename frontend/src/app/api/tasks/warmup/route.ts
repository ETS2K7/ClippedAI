import { NextResponse } from "next/server";

export async function POST() {
  const modalEndpoint = "https://santacruz123-2005--clippedai-clippedai-warmup.modal.run";
  const authToken = process.env.PROCESS_VIDEO_ENDPOINT_AUTH;

  if (!authToken) {
    return NextResponse.json({ error: "Auth token not configured" }, { status: 500 });
  }

  try {
    const response = await fetch(modalEndpoint, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${authToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      return NextResponse.json({ error: "Failed to warm up backend" }, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Warmup proxy error:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
