'use client';

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';
import { NAV_LINKS } from '~/lib/landing/constants';

function ChevronDown({ className = '' }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="none" viewBox="0 0 16 16" className={className}>
      <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="m4 6 4 4 4-4" />
    </svg>
  );
}

function ClippedAiLogo() {
  return (
    <div className="flex items-center gap-2">
      {/* Custom scissors/clip icon — not Opus's play button */}
      <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #8B5CF6, #6366F1)' }}>
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24">
          <path stroke="white" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="m7 4 10 16M17 4 7 20" />
          <circle cx="7" cy="4" r="2" fill="white" opacity="0.6" />
          <circle cx="17" cy="4" r="2" fill="white" opacity="0.6" />
        </svg>
      </div>
      <span className="text-[18px] font-bold tracking-[-0.02em] text-white">
        Clipped<span className="text-violet-400">AI</span>
      </span>
    </div>
  );
}

interface DropdownProps {
  label: string;
  items: { label: string; description?: string; href: string }[];
  isOpen: boolean;
  onToggle: () => void;
}

function NavDropdown({ label, items, isOpen, onToggle }: DropdownProps) {
  return (
    <div className="relative">
      <button
        onClick={onToggle}
        className="flex items-center gap-1.5 text-slate-400 hover:text-white transition-colors duration-200 text-[14px] font-medium"
        aria-expanded={isOpen}
      >
        {label}
        <ChevronDown className={`transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            transition={{ duration: 0.2, ease: [0.19, 1, 0.22, 1] }}
            className="absolute top-full left-1/2 -translate-x-1/2 mt-3 w-[240px] border border-slate-700/50 bg-slate-800/95 backdrop-blur-xl rounded-xl p-1.5 shadow-xl z-50"
          >
            {items.map((item) => (
              <Link
                key={item.label}
                href={item.href}
                className="flex flex-col gap-0.5 px-3 py-2.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/5 transition-all duration-200"
              >
                <span className="text-[13px] font-medium text-white">{item.label}</span>
                {item.description && (
                  <span className="text-[12px] text-slate-500">{item.description}</span>
                )}
              </Link>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function Navbar() {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [activeDropdown, setActiveDropdown] = useState<string | null>(null);
  const navRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleScroll = () => setIsScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    if (activeDropdown) {
      const handleClick = () => setActiveDropdown(null);
      document.addEventListener('click', handleClick);
      return () => document.removeEventListener('click', handleClick);
    }
  }, [activeDropdown]);

  useEffect(() => {
    document.body.style.overflow = isMobileMenuOpen ? 'hidden' : '';
    return () => { document.body.style.overflow = ''; };
  }, [isMobileMenuOpen]);

  return (
    <>
      <header
        id="navbar"
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
          isScrolled
            ? 'bg-slate-900/80 backdrop-blur-xl border-b border-slate-700/30'
            : 'bg-transparent'
        }`}
      >
        <div className="max-w-[1200px] mx-auto px-6 md:px-8">
          <nav ref={navRef} className="relative flex items-center justify-between h-16">
            <Link href="/" className="shrink-0 z-10">
              <ClippedAiLogo />
            </Link>

            {/* Desktop nav */}
            <div
              className="hidden md:flex items-center gap-8"
              onClick={(e) => e.stopPropagation()}
            >
              {Object.entries(NAV_LINKS).map(([key, section]) => (
                <NavDropdown
                  key={key}
                  label={section.label}
                  items={section.items}
                  isOpen={activeDropdown === key}
                  onToggle={() => setActiveDropdown(activeDropdown === key ? null : key)}
                />
              ))}
              <Link href="#faq" className="text-slate-400 hover:text-white transition-colors text-[14px] font-medium">
                FAQ
              </Link>
            </div>

            {/* Desktop actions */}
            <div className="hidden md:flex items-center gap-3 z-10">
              <Link
                href="/login"
                className="text-slate-400 hover:text-white transition-colors text-[14px] font-medium px-3 py-2"
              >
                Sign in
              </Link>
              <Link
                href="/signup"
                className="text-[13px] font-semibold text-white px-5 py-2.5 rounded-xl transition-all duration-200 hover:shadow-lg hover:shadow-violet-500/20"
                style={{ background: 'linear-gradient(135deg, #8B5CF6, #6366F1)' }}
              >
                Get started
              </Link>
            </div>

            {/* Mobile toggle */}
            <button
              className="md:hidden p-2 z-10 text-slate-400"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              aria-label="Toggle menu"
            >
              {isMobileMenuOpen ? (
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24">
                  <path stroke="currentColor" strokeLinecap="round" strokeWidth="2" d="m6 6 12 12M6 18 18 6" />
                </svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24">
                  <path stroke="currentColor" strokeLinecap="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              )}
            </button>
          </nav>
        </div>
      </header>

      {/* Mobile menu */}
      <AnimatePresence>
        {isMobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 md:hidden"
          >
            <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setIsMobileMenuOpen(false)} />
            <motion.nav
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ duration: 0.3, ease: [0.19, 1, 0.22, 1] }}
              className="absolute top-0 right-0 w-[80%] max-w-[360px] h-full bg-slate-900 border-l border-slate-700/50 overflow-y-auto"
            >
              <div className="flex items-center justify-between p-5 border-b border-slate-700/50">
                <Link href="/" onClick={() => setIsMobileMenuOpen(false)}>
                  <ClippedAiLogo />
                </Link>
                <button
                  onClick={() => setIsMobileMenuOpen(false)}
                  className="p-2 text-slate-500 hover:text-white rounded-lg"
                  aria-label="Close menu"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24">
                    <path stroke="currentColor" strokeLinecap="round" strokeWidth="2" d="m6 6 12 12M6 18 18 6" />
                  </svg>
                </button>
              </div>

              <div className="p-5 space-y-1">
                {Object.entries(NAV_LINKS).map(([, section]) =>
                  section.items.map((item) => (
                    <Link
                      key={item.label}
                      href={item.href}
                      className="block px-3 py-3 text-slate-400 hover:text-white transition-colors text-[15px] font-medium rounded-lg hover:bg-white/5"
                      onClick={() => setIsMobileMenuOpen(false)}
                    >
                      {item.label}
                    </Link>
                  ))
                )}
              </div>

              <div className="p-5 space-y-3 border-t border-slate-700/50">
                <Link
                  href="/login"
                  className="block w-full py-3 text-center text-slate-400 hover:text-white transition-colors font-medium rounded-xl border border-slate-700/50"
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  Sign in
                </Link>
                <Link
                  href="/signup"
                  className="block w-full py-3 text-center text-white font-semibold rounded-xl"
                  style={{ background: 'linear-gradient(135deg, #8B5CF6, #6366F1)' }}
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  Get started
                </Link>
              </div>
            </motion.nav>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Spacer */}
      <div className="h-16" />
    </>
  );
}
