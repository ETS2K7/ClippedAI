"use client";
import React from "react";
import { cn } from "~/lib/utils";

export function MovingBorderButton({
  children,
  className,
  containerClassName,
  onClick,
}: {
  children: React.ReactNode;
  className?: string;
  containerClassName?: string;
  onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "relative p-[1px] overflow-hidden rounded-full font-semibold max-w-fit block",
        containerClassName
      )}
    >
      <span className="absolute inset-[-1000%] spin-slow bg-[conic-gradient(from_90deg_at_50%_50%,#000000_0%,#ffffff_50%,#000000_100%)] opacity-70" />
      <span
        className={cn(
          "relative flex items-center justify-center w-full h-full px-8 py-4 bg-black rounded-full text-white text-base md:text-lg font-poppins hover:bg-neutral-900 transition-colors",
          className
        )}
      >
        {children}
      </span>
    </button>
  );
}
