import React, { useRef } from "react";

interface DynamicVideoPlayerProps {
  src: string;
  poster?: string;
  autoPlay?: boolean;
  muted?: boolean;
  loop?: boolean;
  className?: string;
}

const DynamicVideoPlayer: React.FC<DynamicVideoPlayerProps> = ({
  src,
  poster,
  autoPlay = false,
  muted = false,
  loop = false,
  className = "",
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // Directly assign src to <video> element instead of <source> tags to
  // bypass Safari/React hydration conflicts on hot module reloading.
  return (
    <div className={`relative overflow-hidden w-full h-full ${className}`}>
      <video
        ref={videoRef}
        src={src}
        controls
        autoPlay={autoPlay}
        muted={muted}
        loop={loop}
        playsInline
        {...(poster ? { poster } : {})}
        preload="auto"
        className="absolute inset-0 h-full w-full bg-black object-contain"
        tabIndex={0}
        aria-label="Video player"
      >
        Your browser does not support the video tag.
      </video>
    </div>
  );
};

export default DynamicVideoPlayer;
