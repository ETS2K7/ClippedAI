/**
 * Shared task helpers used by task API routes.
 * Single source of truth for status mapping, title derivation, and source type detection.
 */

export function mapStatus(status: string): string {
  switch (status) {
    case "queued":
    case "uploading":
    case "processing":
      return "generating_clips";
    case "processed":
    case "completed":
      return "completed";
    case "failed":
    case "no credits":
      return "failed";
    default:
      return "completed";
  }
}

/** Derive a human-readable title from a file record. */
export function getSourceTitle(file: {
  displayName?: string | null;
  s3Key: string;
}): string {
  if (file.displayName) return file.displayName;
  const parts = file.s3Key.split("/");
  if (parts[0] === "youtube-downloads") {
    // Search segments for a valid 11-char YouTube ID (common in both legacy and new formats)
    const videoId = parts.find(p => p.length === 11);
    if (videoId) return videoId;
  }
  // Uploaded files: <uuid>/original.mp4 — fall back to folder name
  return parts[0] ?? "Video";
}

/** Derive source type from the S3 key prefix. */
export function getSourceType(s3Key: string): "youtube" | "upload" {
  return s3Key.startsWith("youtube-downloads/") ? "youtube" : "upload";
}
