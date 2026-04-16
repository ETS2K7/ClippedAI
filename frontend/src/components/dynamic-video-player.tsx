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

  return (
    <div className={`relative overflow-hidden ${className}`}>
      <video
        ref={videoRef}
        controls
        autoPlay={autoPlay}
        muted={muted}
        loop={loop}
        playsInline
        {...(poster ? { poster } : {})}
        preload="metadata"
        className="absolute inset-0 h-full w-full bg-black object-contain"
        tabIndex={0}
        aria-label="Video player"
      >
        <source src={src} type="video/mp4" />
        Your browser does not support the video tag.
      </video>
    </div>
  );
};

export default DynamicVideoPlayer;
