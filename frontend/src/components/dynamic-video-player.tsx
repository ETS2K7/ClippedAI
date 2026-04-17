import React, { forwardRef, useEffect } from "react";

interface DynamicVideoPlayerProps {
  src: string;
  poster?: string;
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
      poster,
      autoPlay = false,
      muted = false,
      loop = false,
      className = "",
      onPlay,
      onPause,
    },
    ref,
  ) => {
    // Directly assign src to <video> element instead of <source> tags to
    // bypass Safari/React hydration conflicts on hot module reloading.
    return (
      <div className={`relative overflow-hidden w-full h-full ${className}`}>
        <video
          ref={ref}
          src={src}
          controls
          autoPlay={autoPlay}
          muted={muted}
          loop={loop}
          playsInline
          {...(poster ? { poster, loading: "lazy" as const } : {})}
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
