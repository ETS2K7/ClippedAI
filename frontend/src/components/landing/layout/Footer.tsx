'use client';

import Link from 'next/link';
import Image from 'next/image';
import { FOOTER_LINKS, SOCIAL_LINKS } from '~/lib/landing/constants';

function YouTubeIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24">
      <path fill="currentColor" fillRule="evenodd" d="M20.936 6.027c.31.312.533.7.646 1.126C22 8.723 22 12 22 12s0 3.275-.418 4.845a2.52 2.52 0 0 1-1.768 1.78c-1.564.42-7.814.42-7.814.42s-6.25 0-7.814-.42a2.51 2.51 0 0 1-1.768-1.78C2 15.274 2 11.999 2 11.999s0-3.275.418-4.846a2.51 2.51 0 0 1 1.768-1.78C5.75 4.954 12 4.954 12 4.954s6.25 0 7.814.42c.424.116.811.342 1.122.654m-5.754 5.972L9.955 9.025v5.948z" clipRule="evenodd"/>
    </svg>
  );
}

function TikTokIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24">
      <path fill="currentColor" d="M16.227 2h-3.37v13.623c0 1.623-1.296 2.957-2.91 2.957-1.613 0-2.91-1.334-2.91-2.957 0-1.594 1.268-2.898 2.824-2.956v-3.42c-3.428.057-6.194 2.869-6.194 6.376C3.667 19.16 6.49 22 9.976 22s6.309-2.87 6.309-6.377V8.638a7.8 7.8 0 0 0 4.465 1.507v-3.42C18.215 6.638 16.227 4.55 16.227 2"/>
    </svg>
  );
}

function XIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24">
      <path fill="currentColor" d="M17.272 3.71h2.81l-6.14 7.02 7.224 9.553H15.51l-4.431-5.794-5.07 5.794H3.195l6.57-7.509-6.932-9.063h5.8l4.006 5.295zm-.987 14.89h1.558L7.788 5.305H6.116z"/>
    </svg>
  );
}

function DiscordIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24">
      <path fill="currentColor" d="M18.93 5.643a16.5 16.5 0 0 0-4.07-1.263.06.06 0 0 0-.066.03c-.175.314-.37.721-.506 1.042a15.2 15.2 0 0 0-4.573 0A11 11 0 0 0 9.2 4.411a.06.06 0 0 0-.065-.031 16.5 16.5 0 0 0-4.07 1.263.06.06 0 0 0-.028.023c-2.593 3.873-3.303 7.652-2.954 11.383a.07.07 0 0 0 .026.047 16.6 16.6 0 0 0 4.994 2.524.06.06 0 0 0 .07-.023q.58-.788 1.022-1.661a.063.063 0 0 0-.035-.088 11 11 0 0 1-1.56-.744.064.064 0 0 1-.007-.106q.159-.118.31-.243a.06.06 0 0 1 .065-.009c3.273 1.494 6.817 1.494 10.051 0a.06.06 0 0 1 .066.008q.15.124.31.244a.064.064 0 0 1-.005.106q-.747.435-1.561.743a.064.064 0 0 0-.034.089c.3.581.643 1.135 1.02 1.66a.06.06 0 0 0 .07.024 16.5 16.5 0 0 0 5.003-2.524.06.06 0 0 0 .026-.046c.417-4.314-.699-8.062-2.957-11.384a.05.05 0 0 0-.026-.023M8.684 14.777c-.985 0-1.797-.905-1.797-2.016s.796-2.015 1.797-2.015c1.01 0 1.813.912 1.798 2.015 0 1.111-.796 2.016-1.798 2.016m6.646 0c-.985 0-1.797-.905-1.797-2.016s.796-2.015 1.797-2.015c1.009 0 1.813.912 1.797 2.015 0 1.111-.788 2.016-1.797 2.016"/>
    </svg>
  );
}

function LinkedInIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24">
      <path fill="currentColor" d="M20.52 2H3.477C2.66 2 2 2.645 2 3.441v17.114C2 21.352 2.66 22 3.477 22H20.52c.816 0 1.48-.648 1.48-1.441V3.44C22 2.645 21.336 2 20.52 2M7.934 19.043h-2.97V9.496h2.97zM6.449 8.195a1.72 1.72 0 1 1-.006-3.439 1.72 1.72 0 0 1 .006 3.44m12.594 10.848h-2.965v-4.64c0-1.106-.02-2.532-1.543-2.532-1.543 0-1.777 1.207-1.777 2.453v4.719H9.797V9.496h2.844v1.305h.039c.394-.75 1.363-1.543 2.804-1.543 3.004 0 3.559 1.976 3.559 4.547z"/>
    </svg>
  );
}

function InstagramIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24">
      <path fill="currentColor" d="M17.33 7.852a1.2 1.2 0 1 1-.005-2.4 1.2 1.2 0 0 1 .004 2.4"/>
      <path fill="currentColor" fillRule="evenodd" d="M12.01 17.134a5.134 5.134 0 1 1-.02-10.268 5.134 5.134 0 0 1 .02 10.268m-.017-8.467a3.333 3.333 0 1 0 .013 6.667 3.333 3.333 0 0 0-.013-6.667" clipRule="evenodd"/>
      <path fill="currentColor" fillRule="evenodd" d="M5.433 2.539c.635-.25 1.361-.42 2.425-.47 1.067-.05 1.408-.063 4.123-.068s3.056.005 4.123.053c1.064.046 1.791.214 2.427.46a4.9 4.9 0 0 1 1.774 1.15c.557.554.9 1.111 1.156 1.768.249.636.42 1.362.47 2.427.05 1.066.063 1.406.068 4.122.005 2.715-.006 3.056-.053 4.123-.046 1.064-.213 1.791-.46 2.427a4.9 4.9 0 0 1-1.15 1.774c-.554.557-1.11.9-1.768 1.157-.636.248-1.362.418-2.426.47-1.067.05-1.407.062-4.123.067-2.715.005-3.056-.005-4.122-.052s-1.792-.214-2.428-.46a4.9 4.9 0 0 1-1.774-1.15 4.9 4.9 0 0 1-1.156-1.77c-.25-.635-.419-1.361-.47-2.425-.05-1.067-.063-1.408-.068-4.123s.006-3.056.052-4.122c.047-1.065.214-1.792.46-2.429a4.9 4.9 0 0 1 1.15-1.773 4.9 4.9 0 0 1 1.77-1.156m.685 17.268c.353.135.882.297 1.857.34 1.055.046 1.371.056 4.04.05s2.987-.016 4.041-.066c.974-.046 1.504-.21 1.856-.347.466-.182.799-.4 1.148-.75.35-.351.566-.684.746-1.151.136-.354.298-.883.34-1.858.047-1.054.057-1.37.051-4.04-.005-2.67-.016-2.987-.066-4.041-.046-.975-.21-1.504-.347-1.855-.183-.468-.4-.8-.75-1.15a3.1 3.1 0 0 0-1.152-.745c-.353-.137-.883-.298-1.857-.34-1.055-.047-1.371-.056-4.041-.051s-2.985.016-4.04.066c-.975.046-1.503.21-1.856.347-.466.183-.799.4-1.148.75-.35.351-.566.684-.746 1.152-.136.353-.299.883-.34 1.857-.047 1.055-.056 1.371-.051 4.04s.016 2.987.066 4.04c.045.976.21 1.505.347 1.857.183.467.4.8.75 1.149.351.348.685.565 1.152.746" clipRule="evenodd"/>
    </svg>
  );
}

const socialIcons: Record<string, () => JSX.Element> = {
  YouTube: YouTubeIcon,
  TikTok: TikTokIcon,
  Twitter: XIcon,
  Discord: DiscordIcon,
  LinkedIn: LinkedInIcon,
  Instagram: InstagramIcon,
};

export default function Footer() {
  const footerSections = Object.values(FOOTER_LINKS);

  return (
    <footer className="pb-8 bg-background">
      <div className="padding-global">
        <div className="container-1200">
          {/* Main footer card — matches opus.pro: dark rounded card, no logo at top */}
          <div
            className="rounded-[24px] px-10 pt-10 pb-8"
            style={{ background: '#141414' }}
          >
            {/* Links grid — 5 columns directly, NO logo row */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-x-6 gap-y-8 mb-10">
              {footerSections.map((section) => (
                <div key={section.title}>
                  <h3 className="text-[13px] font-semibold text-white mb-4 tracking-normal">
                    {section.title}
                  </h3>
                  <ul className="space-y-[10px]">
                    {section.links.map((link) => (
                      <li key={link.label}>
                        <Link
                          href={link.href}
                          className="text-[13px] text-[#888888] hover:text-white transition-colors duration-200 inline-flex items-center gap-2 leading-snug"
                        >
                          {link.label}
                          {'badge' in link && link.badge && (
                            <span
                              className={`text-[10px] font-semibold px-[6px] py-[1px] rounded-[4px] ${
                                (link.badge as string) === 'Free'
                                  ? 'bg-transparent text-white border border-white/30'
                                  : 'bg-[#6723ff]/20 text-[#9f67ff]'
                              }`}
                            >
                              {link.badge as string}
                            </span>
                          )}
                        </Link>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>

            {/* Badges row + Language selector */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <Image
                  src="/images/soc2-badge.webp"
                  alt="SOC2 TYPE 2 Certified"
                  width={44}
                  height={44}
                  className="opacity-90"
                />
                <Image
                  src="/images/best-software-badge.png"
                  alt="Top 50 AI Products"
                  width={44}
                  height={44}
                  className="opacity-90"
                />
              </div>

              {/* Language selector */}
              <button className="flex items-center gap-[6px] text-[13px] text-[#888888] hover:text-white transition-colors duration-200">
                <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" fill="none" viewBox="0 0 18 19">
                  <path fill="currentColor" fillRule="evenodd" d="M9 2.995a6.825 6.825 0 1 0 0 13.65 6.825 6.825 0 0 0 0-13.65M.825 9.82a8.175 8.175 0 1 1 16.35 0 8.175 8.175 0 0 1-16.35 0" clipRule="evenodd"/>
                  <path fill="currentColor" fillRule="evenodd" d="M1.541 7.064a.45.45 0 0 1 .561-.3l-.13.43.13-.43h.001l.005.002.024.006q.03.01.096.027.128.037.38.1c.335.082.824.192 1.439.303 1.23.221 2.96.443 4.953.443 1.992 0 3.722-.222 4.953-.443a23 23 0 0 0 1.82-.403l.095-.027.024-.006.005-.002.13.43.13.431-.003.001-.007.003-.029.008-.107.03q-.14.04-.404.105c-.35.087-.859.201-1.495.315-1.27.229-3.055.458-5.112.458s-3.842-.229-5.112-.458a24 24 0 0 1-1.9-.42l-.106-.03-.029-.008-.008-.003h-.002l.13-.431-.13.43a.45.45 0 0 1-.302-.56m14.357-.3a.45.45 0 0 1 .26.861l-.13-.43" clipRule="evenodd"/>
                </svg>
                English
                <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" fill="none" viewBox="0 0 16 16">
                  <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="m4 6 4 4 4-4"/>
                </svg>
              </button>
            </div>

            {/* Divider */}
            <div className="h-px bg-white/8 mb-6" />

            {/* Bottom bar: copyright + social icons */}
            <div className="flex flex-col md:flex-row items-center justify-between gap-4">
              <div className="text-[13px] text-[#555555]">
                © {new Date().getFullYear()} ClippedAI by ClippedAI Inc. All rights reserved.
              </div>

              {/* Social icons */}
              <div className="flex items-center gap-[18px]">
                {SOCIAL_LINKS.map((social) => {
                  const IconComponent = socialIcons[social.name];
                  return (
                    <Link
                      key={social.name}
                      href={social.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      aria-label={social.name}
                      className="text-[#666666] hover:text-white transition-colors duration-200"
                    >
                      {IconComponent && <IconComponent />}
                    </Link>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
