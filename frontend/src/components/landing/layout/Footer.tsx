'use client';

import Link from 'next/link';
import { FOOTER_LINKS, SOCIAL_LINKS } from '~/lib/landing/constants';

function InstagramIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z" />
    </svg>
  );
}

function XIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
      <path d="M17.272 3.71h2.81l-6.14 7.02 7.224 9.553H15.51l-4.431-5.794-5.07 5.794H3.195l6.57-7.509-6.932-9.063h5.8l4.006 5.295zm-.987 14.89h1.558L7.788 5.305H6.116z" />
    </svg>
  );
}

const socialIcons: Record<string, () => React.JSX.Element> = {
  Twitter: XIcon,
  Instagram: InstagramIcon,
};

export default function Footer() {
  const sections = Object.values(FOOTER_LINKS);

  return (
    <footer className="border-t border-slate-800/50 py-12 md:py-16">
      <div className="max-w-[1200px] mx-auto px-6 md:px-8">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-12">
          {/* Brand column */}
          <div className="col-span-2 md:col-span-1">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #8B5CF6, #6366F1)' }}>
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24">
                  <path stroke="white" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="m7 4 10 16M17 4 7 20" />
                  <circle cx="7" cy="4" r="2" fill="white" opacity="0.6" />
                  <circle cx="17" cy="4" r="2" fill="white" opacity="0.6" />
                </svg>
              </div>
              <span className="text-[16px] font-bold text-white">
                Clipped<span className="text-violet-400">AI</span>
              </span>
            </div>
            <p className="text-[13px] text-slate-500 leading-[1.6] max-w-[240px]">
              AI-powered video clipping. Turn long videos into viral shorts automatically.
            </p>
          </div>

          {/* Link columns */}
          {sections.map((section) => (
            <div key={section.title}>
              <h3 className="text-[13px] font-semibold text-slate-300 mb-4">
                {section.title}
              </h3>
              <ul className="space-y-2.5">
                {section.links.map((link) => (
                  <li key={link.label}>
                    <Link
                      href={link.href}
                      className="text-[13px] text-slate-500 hover:text-white transition-colors duration-200"
                    >
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Bottom bar */}
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 pt-8 border-t border-slate-800/50">
          <div className="text-[12px] text-slate-600">
            © {new Date().getFullYear()} ClippedAI. All rights reserved.
          </div>
          <div className="flex items-center gap-4">
            {SOCIAL_LINKS.map((social) => {
              const Icon = socialIcons[social.name];
              return (
                <Link
                  key={social.name}
                  href={social.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={social.name}
                  className="text-slate-600 hover:text-white transition-colors duration-200"
                >
                  {Icon && <Icon />}
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </footer>
  );
}
