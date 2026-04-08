import React from 'react';
import { cn } from '~/lib/utils';

export const GridBackground = ({ children, className }: { children?: React.ReactNode, className?: string }) => {
  return (
    <div className={cn("relative w-full overflow-hidden bg-black", className)}>
      <div className="absolute inset-0 bg-[radial-gradient(#222_1px,transparent_1px)] [background-size:24px_24px] [mask-image:radial-gradient(ellipse_60%_60%_at_50%_0%,#000_10%,transparent_100%)] opacity-50 pointer-events-none" />
      <div className="relative z-10 w-full">
        {children}
      </div>
    </div>
  );
};
