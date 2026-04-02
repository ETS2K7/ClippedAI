/**
 * Canonical file processing statuses.
 * Used across the frontend to avoid magic strings.
 * Must match values stored in the database.
 */
export const FileStatus = {
  QUEUED: "queued",
  UPLOADING: "uploading",
  PROCESSING: "processing",
  PROCESSED: "processed",
  FAILED: "failed",
} as const;

export type FileStatusType = (typeof FileStatus)[keyof typeof FileStatus];
