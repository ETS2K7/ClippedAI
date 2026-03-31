'use client';

import { motion } from 'framer-motion';

interface ScrollRevealProps {
  children: React.ReactNode;
  direction?: 'up' | 'down' | 'left' | 'right';
  delay?: number;
  duration?: number;
  className?: string;
  once?: boolean;
}

export default function ScrollReveal({
  children,
  direction = 'up',
  delay = 0,
  duration = 0.7,
  className = '',
  once = true,
}: ScrollRevealProps) {
  const getInitialTransform = () => {
    switch (direction) {
      case 'up': return { y: 40 };
      case 'down': return { y: -40 };
      case 'left': return { x: 40 };
      case 'right': return { x: -40 };
      default: return { y: 40 };
    }
  };

  return (
    <motion.div
      className={className}
      initial={{
        opacity: 0,
        ...getInitialTransform(),
      }}
      whileInView={{
        opacity: 1,
        y: 0,
        x: 0,
      }}
      transition={{
        duration,
        delay,
        ease: [0.19, 1, 0.22, 1],
      }}
      viewport={{ once, margin: '-50px' }}
    >
      {children}
    </motion.div>
  );
}
