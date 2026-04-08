"use client";
import React from 'react';
import { motion } from 'framer-motion';
import { cn } from '~/lib/utils';

export const Spotlight = ({ className, fill = "white" }: { className?: string, fill?: string }) => {
  return (
    <div className={cn("pointer-events-none absolute inset-0 z-0 h-full w-full overflow-hidden block mask-image-linear-vertical", className)}>
      <motion.div
        animate={{
          x: [0, 30, 0],
          y: [0, 20, 0],
          opacity: [0.15, 0.4, 0.15],
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: "easeInOut",
        }}
        className="absolute -top-40 -left-20 w-[150%] max-w-3xl h-[40rem] md:h-[60rem] rounded-[100%] blur-[100px] opacity-20 transform-gpu"
        style={{
          background: `radial-gradient(ellipse at center, ${fill} 0%, transparent 60%)`,
        }}
      />
      <motion.div
        animate={{
          x: [0, -40, 0],
          y: [0, -30, 0],
          opacity: [0.1, 0.25, 0.1],
        }}
        transition={{
          duration: 15,
          repeat: Infinity,
          ease: "easeInOut",
        }}
        className="absolute top-1/4 -right-10 w-full max-w-2xl h-[30rem] md:h-[50rem] rounded-[100%] blur-[100px] opacity-10 transform-gpu"
        style={{
          background: `radial-gradient(ellipse at center, ${fill} 0%, transparent 60%)`,
        }}
      />
    </div>
  );
};
