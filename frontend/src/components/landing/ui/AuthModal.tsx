'use client';

import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';

interface AuthModalProps {
  onClose: () => void;
}

function GoogleIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 48 48" className="shrink-0">
      <path fill="#FFC107" d="M43.611,20.083H42V20H24v8h11.303c-1.649,4.657-6.08,8-11.303,8c-6.627,0-12-5.373-12-12c0-6.627,5.373-12,12-12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C34.046,6.053,29.268,4,24,4C12.955,4,4,12.955,4,24c0,11.045,8.955,20,20,20c11.045,0,20-8.955,20-20C44,22.659,43.862,21.35,43.611,20.083z"/>
      <path fill="#FF3D00" d="M6.306,14.691l6.571,4.819C14.655,15.108,18.961,12,24,12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C34.046,6.053,29.268,4,24,4C16.318,4,9.656,8.337,6.306,14.691z"/>
      <path fill="#4CAF50" d="M24,44c5.166,0,9.86-1.977,13.409-5.192l-6.19-5.238C29.211,35.091,26.715,36,24,36c-5.202,0-9.619-3.317-11.283-7.946l-6.522,5.025C9.505,39.556,16.227,44,24,44z"/>
      <path fill="#1976D2" d="M43.611,20.083H42V20H24v8h11.303c-0.792,2.237-2.231,4.166-4.087,5.571c0.001-0.001,0.002-0.001,0.003-0.002l6.19,5.238C36.971,39.205,44,34,44,24C44,22.659,43.862,21.35,43.611,20.083z"/>
    </svg>
  );
}

function AppleIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 384 512" fill="white" className="shrink-0 mb-[3px]">
      <path d="M318.7 268.7c-.2-36.7 16.4-64.4 50-84.8-18.8-26.9-47.2-41.7-84.7-44.6-35.5-2.8-74.3 20.7-88.5 20.7-15 0-49.4-19.7-76.4-19.7C63.3 141.2 4 184.8 4 273.5q0 39.3 14.4 81.2c12.8 36.7 59 126.7 107.2 125.2 25.2-.6 43-17.9 75.8-17.9 31.8 0 48.3 17.9 76.4 17.9 48.6-.7 90.4-82.5 102.6-119.3-65.2-30.7-61.7-90-61.7-91.9zm-56.6-164.2c27.3-32.4 24.8-61.9 24-72.5-24.1 1.4-52 16.4-67.9 34.9-17.5 19.8-27.8 44.3-25.6 71.9 26.1 2 49.9-11.4 69.5-34.3z"/>
    </svg>
  );
}

export default function AuthModal({ onClose }: AuthModalProps) {
  const [email, setEmail] = useState('');

  // Close modal when pressing Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center p-4">
      {/* Blurred Backdrop */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.2 }}
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose} // clicking outside closes modal
      />

      {/* Modal Container */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 10 }}
        transition={{ duration: 0.3, ease: [0.19, 1, 0.22, 1] }}
        className="relative w-full max-w-[440px] bg-[#222224] rounded-[24px] p-8 md:px-10 md:py-9 shadow-2xl z-10"
        onClick={(e) => e.stopPropagation()} // prevent closing when clicking inside
      >
        <div className="flex flex-col items-center text-center">
          
          {/* Header */}
          <h2 className="text-[26px] md:text-[28px] font-bold text-white mb-2 leading-tight tracking-tight">
            Finish signing up to get your<br />free clips
          </h2>
          <p className="text-[15px] text-[#888888] font-medium mb-7">
            Free plan available. No credit card required.
          </p>

          {/* Social Sign In Buttons */}
          <div className="flex flex-col gap-3 w-full mb-6 relative">
            <button className="relative w-full flex items-center justify-center bg-[#363638] hover:bg-[#3f3f41] transition-colors rounded-[12px] py-[14px]">
              <span className="absolute left-4">
                <GoogleIcon />
              </span>
              <span className="text-[15px] font-semibold text-white tracking-wide">
                Continue with Google
              </span>
            </button>
            <button className="relative w-full flex items-center justify-center bg-[#363638] hover:bg-[#3f3f41] transition-colors rounded-[12px] py-[14px]">
              <span className="absolute left-4">
                <AppleIcon />
              </span>
              <span className="text-[15px] font-semibold text-white tracking-wide">
                Continue with Apple
              </span>
            </button>
          </div>

          {/* Divider */}
          <div className="flex items-center w-full gap-4 mb-6">
            <div className="flex-1 h-px bg-white/10" />
            <span className="text-[#555555] text-[13px] font-medium uppercase tracking-wide">or continue with email</span>
            <div className="flex-1 h-px bg-white/10" />
          </div>

          {/* Email Form */}
          <form className="w-full flex flex-col gap-3 mb-6" onSubmit={(e) => e.preventDefault()}>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Enter email address"
              required
              className="w-full bg-transparent border border-white/10 hover:border-white/30 focus:border-white/50 focus:outline-none transition-colors rounded-[12px] px-4 py-[14px] text-[15px] text-white placeholder:text-[#555555] font-medium"
            />
            <button type="submit" className="w-full bg-white hover:bg-white/90 transition-colors text-black font-semibold text-[15px] rounded-[12px] py-[14px] mt-1 tracking-wide">
              Continue with email
            </button>
          </form>

          {/* Facebook Logic Helper */}
          <p className="text-[14px] text-[#888888] mb-8 font-medium">
            Facebook user? Login <button className="text-white hover:underline underline-offset-2 ml-1 cursor-pointer">here</button>
          </p>

          {/* Footer Terms */}
          <p className="text-[12px] text-[#555555] max-w-[300px] leading-relaxed mx-auto">
            By continuing, you agree to ClippedAI&apos;s <button className="underline hover:text-[#888888] transition-colors cursor-pointer">Terms of Service</button>.
            <br />
            Read our <button className="underline hover:text-[#888888] transition-colors cursor-pointer">Privacy Policy</button>.
          </p>
          
        </div>
      </motion.div>
    </div>
  );
}
