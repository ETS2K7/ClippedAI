'use client';

import React from 'react';
import { LayoutGroup } from 'framer-motion';

export default function SharedLayoutProvider({ children }: { children: React.ReactNode }) {
  return <LayoutGroup>{children}</LayoutGroup>;
}
