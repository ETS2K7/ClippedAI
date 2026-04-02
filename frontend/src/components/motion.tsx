"use client";

import { motion, type Variants, type Transition } from "framer-motion";
import React from "react";

/* ═══════════════════════════════════════════════════════════════════
   Midnight Forge — Motion Design System
   
   Spring-physics animations, staggered reveals, magnetic hover,
   and scroll-triggered effects for premium interactive feel.
   ═══════════════════════════════════════════════════════════════════ */

// ── Shared Easing & Spring Configs ─────────────────────────────────

export const EASE_OUT_EXPO: Transition["ease"] = [0.16, 1, 0.3, 1];
export const EASE_OUT_QUINT: Transition["ease"] = [0.22, 1, 0.36, 1];

export const SPRING_GENTLE: Transition = {
  type: "spring",
  stiffness: 100,
  damping: 15,
  mass: 0.8,
};

export const SPRING_SNAPPY: Transition = {
  type: "spring",
  stiffness: 300,
  damping: 25,
  mass: 0.5,
};

export const SPRING_BOUNCY: Transition = {
  type: "spring",
  stiffness: 400,
  damping: 17,
  mass: 0.6,
};

// ── Variant Libraries ──────────────────────────────────────────────

/** Fade up on mount — use for page-level content */
export const fadeInUp: Variants = {
  hidden: { opacity: 0, y: 20 },
  visible: (i = 0) => ({
    opacity: 1,
    y: 0,
    transition: {
      delay: i * 0.06,
      duration: 0.5,
      ease: EASE_OUT_EXPO,
    },
  }),
};

/** Fade in from scale — use for cards appearing */
export const fadeInScale: Variants = {
  hidden: { opacity: 0, scale: 0.96, y: 10 },
  visible: (i = 0) => ({
    opacity: 1,
    scale: 1,
    y: 0,
    transition: {
      delay: i * 0.08,
      duration: 0.45,
      ease: EASE_OUT_EXPO,
    },
  }),
};

/** Slide in from left — use for sidebars, panels */
export const slideInLeft: Variants = {
  hidden: { opacity: 0, x: -30 },
  visible: {
    opacity: 1,
    x: 0,
    transition: { duration: 0.5, ease: EASE_OUT_EXPO },
  },
};

/** Slide in from right — use for action bars, notifications */
export const slideInRight: Variants = {
  hidden: { opacity: 0, x: 30 },
  visible: {
    opacity: 1,
    x: 0,
    transition: { duration: 0.5, ease: EASE_OUT_EXPO },
  },
};

/** Pop up from bottom — use for command bars, toasts */
export const popUp: Variants = {
  hidden: { opacity: 0, y: 40, scale: 0.95 },
  visible: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: SPRING_BOUNCY,
  },
  exit: {
    opacity: 0,
    y: 20,
    scale: 0.95,
    transition: { duration: 0.2, ease: EASE_OUT_QUINT },
  },
};

/** Container variant for staggering children */
export const staggerContainer: Variants = {
  hidden: { opacity: 1 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.06,
      delayChildren: 0.1,
    },
  },
};

export const staggerContainerSlow: Variants = {
  hidden: { opacity: 1 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.12,
      delayChildren: 0.15,
    },
  },
};

// ── Reusable Motion Components ─────────────────────────────────────

/** Page wrapper — fades up content on mount */
export function PageTransition({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: EASE_OUT_EXPO }}
    >
      {children}
    </motion.div>
  );
}

/** Stagger wrapper — automatically staggers children animations */
export function StaggerList({
  children,
  className = "",
  slow = false,
}: {
  children: React.ReactNode;
  className?: string;
  slow?: boolean;
}) {
  return (
    <motion.div
      className={className}
      variants={slow ? staggerContainerSlow : staggerContainer}
      initial="hidden"
      animate="visible"
    >
      {children}
    </motion.div>
  );
}

/** Individual staggered item */
export function StaggerItem({
  children,
  className = "",
  index = 0,
}: {
  children: React.ReactNode;
  className?: string;
  index?: number;
}) {
  return (
    <motion.div
      className={className}
      variants={fadeInUp}
      custom={index}
    >
      {children}
    </motion.div>
  );
}

/** Glass card with spring hover effect */
export function GlassCard({
  children,
  className = "",
  hoverScale = 1.01,
  hoverGlow = true,
  onClick,
}: {
  children: React.ReactNode;
  className?: string;
  hoverScale?: number;
  hoverGlow?: boolean;
  onClick?: () => void;
}) {
  return (
    <motion.div
      className={`glass-card-hover rounded-xl ${className}`}
      whileHover={{
        scale: hoverScale,
        ...(hoverGlow && {
          boxShadow: "0 0 40px oklch(0.65 0.25 290 / 0.08), 0 8px 32px oklch(0 0 0 / 0.3)",
        }),
      }}
      whileTap={{ scale: 0.99 }}
      transition={SPRING_SNAPPY}
      onClick={onClick}
    >
      {children}
    </motion.div>
  );
}

/** Animated button with spring press feedback */
export function SpringButton({
  children,
  className = "",
  onClick,
  disabled = false,
  type = "button",
}: {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
  disabled?: boolean;
  type?: "button" | "submit";
}) {
  return (
    <motion.button
      type={type}
      className={className}
      onClick={onClick}
      disabled={disabled}
      whileHover={disabled ? {} : { scale: 1.02, y: -1 }}
      whileTap={disabled ? {} : { scale: 0.97 }}
      transition={SPRING_SNAPPY}
    >
      {children}
    </motion.button>
  );
}

/** Animated gradient border — wraps children with a rotating gradient border */
export function AnimatedBorder({
  children,
  className = "",
  borderRadius = "0.75rem",
  active = true,
}: {
  children: React.ReactNode;
  className?: string;
  borderRadius?: string;
  active?: boolean;
}) {
  if (!active) {
    return <div className={className}>{children}</div>;
  }

  return (
    <div className={`relative ${className}`} style={{ borderRadius }}>
      {/* Rotating gradient border */}
      <motion.div
        className="absolute -inset-[1px] rounded-[inherit] opacity-60"
        style={{
          background: "conic-gradient(from 0deg, oklch(0.65 0.25 290 / 0.3), transparent, oklch(0.65 0.25 290 / 0.15), transparent, oklch(0.65 0.25 290 / 0.3))",
          borderRadius: "inherit",
        }}
        animate={{ rotate: 360 }}
        transition={{
          repeat: Infinity,
          duration: 6,
          ease: "linear",
        }}
      />
      {/* Inner content */}
      <div className="relative bg-[#0a0a0f] rounded-[inherit]" style={{ borderRadius: "inherit" }}>
        {children}
      </div>
    </div>
  );
}

/** Scroll-triggered reveal for app pages */
export function AppReveal({
  children,
  className = "",
  direction = "up",
  delay = 0,
}: {
  children: React.ReactNode;
  className?: string;
  direction?: "up" | "down" | "left" | "right";
  delay?: number;
}) {
  const directionMap = {
    up: { y: 30 },
    down: { y: -30 },
    left: { x: 30 },
    right: { x: -30 },
  };

  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, ...directionMap[direction] }}
      whileInView={{ opacity: 1, y: 0, x: 0 }}
      transition={{ duration: 0.6, delay, ease: EASE_OUT_EXPO }}
      viewport={{ once: true, margin: "-60px" }}
    >
      {children}
    </motion.div>
  );
}

/** Shimmer loading skeleton with motion */
export function MotionSkeleton({
  className = "",
  width,
  height,
}: {
  className?: string;
  width?: string | number;
  height?: string | number;
}) {
  return (
    <motion.div
      className={`rounded-lg bg-white/[0.06] ${className}`}
      style={{ width, height }}
      animate={{
        opacity: [0.4, 0.7, 0.4],
      }}
      transition={{
        repeat: Infinity,
        duration: 1.8,
        ease: "easeInOut",
      }}
    />
  );
}

/** Number counter animation */
export function AnimatedNumber({
  value,
  className = "",
  suffix = "",
}: {
  value: number;
  className?: string;
  suffix?: string;
}) {
  return (
    <motion.span
      className={className}
      key={value}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: EASE_OUT_EXPO }}
    >
      {value}{suffix}
    </motion.span>
  );
}

/** Presence wrapper for enter/exit animations */
export { AnimatePresence } from "framer-motion";
