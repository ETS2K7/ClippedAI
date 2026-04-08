import React from 'react';
import Link from 'next/link';

export default function FooterSection() {
  return (
    <footer className="w-full bg-black py-16 text-white border-t border-white/5 font-poppins relative overflow-hidden">
      <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />
      <div className="container mx-auto px-4 md:px-6 relative z-10">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12 mb-16">
          <div className="md:col-span-1">
            <Link href="/" className="font-bold text-3xl font-oswald tracking-widest mb-6 block text-white drop-shadow-md">
              CLIPPED
            </Link>
            <p className="text-neutral-500 text-sm leading-relaxed max-w-xs">
              The supreme AI engine for modern video production and automated clipping workflows.
            </p>
          </div>
          
          <div>
            <h4 className="font-semibold text-lg mb-6 text-white tracking-wide">Product</h4>
            <ul className="space-y-4 text-neutral-400 text-sm font-medium">
              <li><a href="#" className="hover:text-white transition-colors">Features</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Pricing</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Supported Formats</a></li>
              <li><a href="#" className="hover:text-white transition-colors">API</a></li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-semibold text-lg mb-6 text-white tracking-wide">Resources</h4>
            <ul className="space-y-4 text-neutral-400 text-sm font-medium">
              <li><a href="#" className="hover:text-white transition-colors">Documentation</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Help Center</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Creator Guide</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Blog</a></li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-semibold text-lg mb-6 text-white tracking-wide">Company</h4>
            <ul className="space-y-4 text-neutral-400 text-sm font-medium">
              <li><a href="#" className="hover:text-white transition-colors">About Us</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Careers</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Privacy Policy</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Terms of Service</a></li>
            </ul>
          </div>
        </div>
        
        <div className="pt-8 border-t border-white/10 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-neutral-500 text-sm font-medium">
            &copy; {new Date().getFullYear()} ClippedAI. All rights reserved.
          </p>
          <div className="flex items-center space-x-8 text-sm text-neutral-500 font-medium">
            <a href="https://github.com/ebelthomasseiko" target="_blank" rel="noreferrer" className="hover:text-white transition-colors">
              GitHub
            </a>
            <a href="#" className="hover:text-white transition-colors">
              Twitter
            </a>
            <a href="#" className="hover:text-white transition-colors">
              Discord
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
