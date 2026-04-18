import React, { forwardRef, useEffect, useRef } from "react";
import Hls from "hls.js";

interface DynamicVideoPlayerProps {
  src: string;
  hlsUrl?: string | null;
  poster?: string;
  thumbnailKeys?: Record<string, string>;
  autoPlay?: boolean;
  muted?: boolean;
  loop?: boolean;
  className?: string;
  onPlay?: () => void;
  onPause?: () => void;
}

const DynamicVideoPlayer = forwardRef<HTMLVideoElement, DynamicVideoPlayerProps>(
  (
    {
      src,
      hlsUrl,
      poster,
      thumbnailKeys,
      autoPlay = false,
      muted = false,
      loop = false,
      className = "",
      onPlay,
      onPause,
    },
    ref,
  ) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const hlsRef = useRef<Hls | null>(null);

    // Forward ref to video element
    useEffect(() => {
      if (typeof ref === "function") {
        ref(videoRef.current);
      } else if (ref) {
        ref.current = videoRef.current;
      }
    }, [ref]);

    // Initialize HLS player if hlsUrl is provided
    useEffect(() => {
      const video = videoRef.current;
      if (!video || !hlsUrl) return;

      console.log("HLS URL:", hlsUrl);
      console.log("HLS supported:", Hls.isSupported());
      console.log("Native HLS supported:", video.canPlayType("application/vnd.apple.mpegurl"));

      if (Hls.isSupported()) {
        const hls = new Hls({
          enableWorker: false,
          lowLatencyMode: false,
          maxBufferLength: 30,
          maxMaxBufferLength: 60,
        });
        hlsRef.current = hls;

        hls.on(Hls.Events.MANIFEST_PARSED, () => {
          console.log("HLS manifest parsed successfully");
        });

        hls.on(Hls.Events.ERROR, (event, data) => {
          console.error("HLS error:", data.type, data.details, data.fatal);
          if (data.fatal) {
            switch (data.type) {
              case Hls.ErrorTypes.NETWORK_ERROR:
                console.log("Attempting to recover from network error");
                hls.startLoad();
                break;
              case Hls.ErrorTypes.MEDIA_ERROR:
                console.log("Attempting to recover from media error");
                hls.recoverMediaError();
                break;
              default:
                console.log("Fatal error, cannot recover");
                break;
            }
          }
        });

        hls.loadSource(hlsUrl);
        hls.attachMedia(video);

        return () => {
          hls.destroy();
        };
      } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
        // Native HLS support (Safari)
        console.log("Using native HLS support");
        video.src = hlsUrl;
      } else {
        console.error("HLS not supported on this device");
      }
    }, [hlsUrl]);

    // Generate srcset from thumbnailKeys if available
    const srcset = thumbnailKeys
      ? Object.entries(thumbnailKeys)
          .map(([size, url]) => {
            const width = size.replace("thumb_", "").replace("w", "");
            return `${url} ${width}w`;
          })
          .join(", ")
      : undefined;

    // Use HLS URL if available, otherwise fall back to direct MP4
    const videoSrc = hlsUrl ? undefined : src;

    return (
      <div className={`relative overflow-hidden w-full h-full ${className}`}>
        <video
          ref={videoRef}
          src={videoSrc}
          controls
          autoPlay={autoPlay}
          muted={muted}
          loop={loop}
          playsInline
          {...(poster || srcset
            ? {
                poster: poster,
                ...(srcset && { "data-srcset": srcset }),
                loading: "lazy" as const,
              }
            : {})}
          preload="metadata"
          className="absolute inset-0 h-full w-full bg-black object-contain"
          tabIndex={0}
          aria-label="Video player"
          onPlay={onPlay}
          onPause={onPause}
        >
          Your browser does not support the video tag.
        </video>
      </div>
    );
  },
);

DynamicVideoPlayer.displayName = "DynamicVideoPlayer";

export default DynamicVideoPlayer;
