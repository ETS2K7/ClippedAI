"use client";

import type { Clip } from "@prisma/client";
import { Download, Loader2, Play } from "lucide-react";
import { useState } from "react";
import { getClipDownloadUrl } from "~/actions/generation";
import { Button } from "./ui/button";

function ClipCard({ clip, videoUrl }: { clip: Clip; videoUrl: string | null }) {
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);

  const handleDownload = async () => {
    if (downloadUrl) {
      triggerDownloadLink(downloadUrl);
      return;
    }

    setIsDownloading(true);
    try {
      const result = await getClipDownloadUrl(clip.id);
      if (result.success && result.url) {
        setDownloadUrl(result.url);
        triggerDownloadLink(result.url);
      } else if (result.error) {
        console.error("Failed to get download url: " + result.error);
      }
    } catch (error) {
      console.error("Failed to fetch clip download URL:", error);
    } finally {
      setIsDownloading(false);
    }
  };

  const triggerDownloadLink = (url: string) => {
    const link = document.createElement("a");
    link.href = url;
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="flex max-w-52 flex-col gap-2">
      <div className="bg-muted">
        {videoUrl ? (
          <video
            src={videoUrl}
            controls
            preload="metadata"
            className="h-full w-full rounded-md object-cover"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center">
            <Play className="text-muted-foreground h-10 w-10 opacity-50" />
          </div>
        )}
      </div>
      <div className="flex flex-col gap-2">
        <Button
          onClick={handleDownload}
          variant="outline"
          size="sm"
          disabled={isDownloading}
        >
          {isDownloading ? (
            <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />
          ) : (
            <Download className="mr-1.5 h-4 w-4" />
          )}
          Download
        </Button>
      </div>
    </div>
  );
}

export function ClipDisplay({ clips, videoUrls }: { clips: Clip[]; videoUrls: (string | null)[] }) {
  if (clips.length === 0) {
    return (
      <p className="text-muted-foreground p-4 text-center">
        No clips generated yet.
      </p>
    );
  }
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
      {clips.map((clip, index) => (
        <ClipCard key={clip.id} clip={clip} videoUrl={videoUrls[index] ?? null} />
      ))}
    </div>
  );
}
