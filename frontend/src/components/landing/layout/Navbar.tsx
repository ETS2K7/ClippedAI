'use client';

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';
import { NAV_LINKS, DASHBOARD_URL } from '~/lib/landing/constants';
import AuthModal from '~/components/landing/ui/AuthModal';

function ChevronDown({ className = '' }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 16 16" className={className}>
      <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="m4 6 4 4 4-4" />
    </svg>
  );
}

function MenuIcon() {
  return (
    <div className="flex flex-col gap-[5px]">
      <span className="block w-[18px] h-[2px] bg-white rounded-full transition-all" />
      <span className="block w-[18px] h-[2px] bg-white rounded-full transition-all" />
      <span className="block w-[18px] h-[2px] bg-white rounded-full transition-all" />
    </div>
  );
}

function CloseIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24">
      <path stroke="currentColor" strokeLinecap="round" strokeWidth="2" d="m6 6 12 12M6 18 18 6" />
    </svg>
  );
}

function ClippedAiLogo() {
  return (
    <div className="flex items-center gap-[6px]">
      <svg width="26" height="26" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path fillRule="evenodd" clipRule="evenodd" d="M16 32C24.8366 32 32 24.8366 32 16C32 7.16344 24.8366 0 16 0C7.16344 0 0 7.16344 0 16C0 24.8366 7.16344 32 16 32ZM21.4706 16.8927C22.0543 16.5126 22.0543 15.4874 21.4706 15.1073L13.8863 10.1558C13.2435 9.73696 12.3871 10.1992 12.3871 11.0485V20.9515C12.3871 21.8008 13.2435 22.263 13.8863 21.8442L21.4706 16.8927Z" fill="currentColor"/>
      </svg>
      <span className="text-[22px] font-bold tracking-[-0.03em]">ClippedAI</span>
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
  const ref = useRef<HTMLDivElement>(null);

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={onToggle}
        className="flex items-center gap-1 text-text-secondary hover:text-white transition-colors duration-200 text-[15px] font-medium whitespace-nowrap"
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
            className="absolute top-full left-1/2 -translate-x-1/2 mt-3 w-[280px] bg-background-secondary
                       border border-border rounded-2xl p-2 shadow-xl z-50"
          >
            {items.map((item) => (
              <Link
                key={item.label}
                href={item.href}
                className="flex flex-col gap-0.5 px-3 py-2.5 rounded-xl
                           text-text-secondary hover:text-white hover:bg-white/5
                           transition-all duration-200"
              >
                <span className="text-sm font-medium text-white">{item.label}</span>
                {item.description && (
                  <span className="text-xs text-text-muted">{item.description}</span>
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
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    const handleClickOutside = () => {
      setActiveDropdown(null);
    };
    if (activeDropdown) {
      document.addEventListener('click', handleClickOutside);
      return () => document.removeEventListener('click', handleClickOutside);
    }
  }, [activeDropdown]);

  useEffect(() => {
    if (isMobileMenuOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => { document.body.style.overflow = ''; };
  }, [isMobileMenuOpen]);

  const toggleDropdown = (key: string) => {
    setActiveDropdown(activeDropdown === key ? null : key);
  };

  return (
    <>
      <header
        id="navbar"
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
          isScrolled
            ? 'bg-background/90 backdrop-blur-md border-b border-white/5'
            : 'bg-transparent'
        }`}
      >
        <div className="w-full px-6 md:px-8">
          <nav className="relative flex items-center justify-between h-[4.5rem]">

            <Link href="/" className="text-white shrink-0 z-10">
                <ClippedAiLogo />
            </Link>

            {/* Desktop Navigation — flex centered between left and right */}
            <div
              className="hidden lg:flex flex-1 items-center justify-center gap-6"
              onClick={(e) => e.stopPropagation()}
            >
              <NavDropdown
                label={NAV_LINKS.features.label}
                items={NAV_LINKS.features.items}
                isOpen={activeDropdown === 'features'}
                onToggle={() => toggleDropdown('features')}
              />
              <NavDropdown
                label={NAV_LINKS.solutions.label}
                items={NAV_LINKS.solutions.items}
                isOpen={activeDropdown === 'solutions'}
                onToggle={() => toggleDropdown('solutions')}
              />
              <NavDropdown
                label={NAV_LINKS.resources.label}
                items={NAV_LINKS.resources.items}
                isOpen={activeDropdown === 'resources'}
                onToggle={() => toggleDropdown('resources')}
              />
            </div>

            {/* Desktop Actions — right */}
            <div className="hidden lg:flex items-center justify-end gap-5 z-10 shrink-0">
              <Link
                href="/login"
                className="text-white hover:text-white/80 transition-colors duration-200 text-[15px] font-medium whitespace-nowrap"
              >
                Sign in
              </Link>
              <Link
                href="/signup"
                className="bg-white text-black hover:bg-white/90 transition-colors duration-200 text-[14px] font-semibold whitespace-nowrap px-4 py-[10px] rounded-[10px]"
              >
                Sign up
              </Link>
            </div>

            {/* Mobile Menu Button */}
            <button
              className="lg:hidden p-2 z-10"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              aria-label="Toggle menu"
            >
              {isMobileMenuOpen ? <CloseIcon /> : <MenuIcon />}
            </button>
          </nav>
        </div>
      </header>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isMobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-40 lg:hidden"
          >
            {/* Backdrop */}
            <div className="absolute inset-0 bg-black/60" onClick={() => setIsMobileMenuOpen(false)} />

            {/* Menu Panel */}
            <motion.nav
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ duration: 0.3, ease: [0.19, 1, 0.22, 1] }}
              className="absolute top-0 right-0 w-[85%] max-w-[400px] h-full bg-background-secondary
                         border-l border-border overflow-y-auto"
            >
              {/* Mobile menu header */}
              <div className="flex items-center justify-between p-5 border-b border-border">
                <Link href="/" className="text-white" onClick={() => setIsMobileMenuOpen(false)}>
                    <ClippedAiLogo />
                </Link>
                <button
                  onClick={() => setIsMobileMenuOpen(false)}
                  className="p-2 text-text-secondary hover:text-white rounded-xl border border-border
                             hover:bg-white/5 transition-all"
                  aria-label="Close menu"
                >
                  <CloseIcon />
                </button>
              </div>

              {/* Mobile menu links */}
              <div className="p-5 space-y-1">
                {Object.entries(NAV_LINKS).map(([key, section]) => (
                  <MobileDropdown
                    key={key}
                    label={section.label}
                    items={section.items}
                    onClose={() => setIsMobileMenuOpen(false)}
                  />
                ))}
                <Link
                  href="#pricing"
                  className="block px-3 py-3 text-text-secondary hover:text-white
                             transition-colors font-medium rounded-xl hover:bg-white/5"
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  Pricing
                </Link>
              </div>

              {/* Mobile menu actions */}
              <div className="p-5 space-y-3 border-t border-border">
                <button
                  className="w-full py-3 text-center text-text-secondary hover:text-white
                             transition-colors font-medium rounded-xl border border-border hover:bg-white/5"
                  onClick={() => {
                    setIsMobileMenuOpen(false);
                    window.location.href = '/login';
                  }}
                >
                  Sign in
                </button>
                <button
                  className="bg-white text-black w-full text-center block rounded-xl py-3 font-semibold hover:bg-white/90 transition-colors"
                  onClick={() => {
                    setIsMobileMenuOpen(false);
                    window.location.href = '/signup';
                  }}
                >
                  Sign up
                </button>
              </div>
            </motion.nav>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {isAuthModalOpen && (
          <AuthModal onClose={() => setIsAuthModalOpen(false)} />
        )}
      </AnimatePresence>

      {/* Spacer */}
      <div className="h-[4.5rem]" />
    </>
  );
}

function MobileDropdown({
  label,
  items,
  onClose,
}: {
  label: string;
  items: { label: string; description?: string; href: string }[];
  onClose: () => void;
}) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between w-full px-3 py-3 text-text-secondary
                   hover:text-white transition-colors font-medium rounded-xl hover:bg-white/5"
      >
        {label}
        <ChevronDown className={`transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="pl-4 space-y-0.5 pb-2">
              {items.map((item) => (
                <Link
                  key={item.label}
                  href={item.href}
                  className="block px-3 py-2.5 text-sm text-text-muted hover:text-white
                             transition-colors rounded-lg hover:bg-white/5"
                  onClick={onClose}
                >
                  {item.label}
                </Link>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
