// ===== Navigation Links =====
export const NAV_LINKS = {
  features: {
    label: 'Features',
    items: [
      { label: 'AI Clip Detection', description: 'Find the best moments automatically', href: '#features' },
      { label: 'Smart Reframing', description: 'Vertical crops with face tracking', href: '#features' },
      { label: 'Auto Subtitles', description: 'Word-synced animated captions', href: '#features' },
    ],
  },
  resources: {
    label: 'Resources',
    items: [
      { label: 'Blog', description: 'Tips and updates', href: '#' },
      { label: 'Help Center', description: 'Guides and FAQ', href: '#faq' },
    ],
  },
};

export const DASHBOARD_URL = '/dashboard';

// ===== Hero Section =====
export const HERO_CONTENT = {
  headline: 'Turn any video into scroll-stopping clips',
  subheadline: 'Upload a video or paste a YouTube link. ClippedAI finds the best moments, reframes for vertical, adds captions, and delivers ready-to-post clips in minutes.',
  ctaPrimary: 'Start clipping — free',
  ctaSecondary: 'See how it works',
};

// ===== Features =====
export const FEATURES = [
  {
    title: 'AI Clip Detection',
    description: 'Our AI analyzes your video — dialogue, energy, pacing, and scene changes — to identify the most engaging moments. No manual trimming required.',
    icon: 'sparkles',
  },
  {
    title: 'Smart Vertical Reframing',
    description: 'Automatically crops your 16:9 video to 9:16 with face-tracking that follows the active speaker. Perfect framing, every time.',
    icon: 'crop',
  },
  {
    title: 'Word-Synced Subtitles',
    description: 'Adds animated, word-by-word captions to every clip. Customizable fonts, colors, and styles to match your brand.',
    icon: 'subtitles',
  },
];

// ===== How It Works =====
export const HOW_IT_WORKS = [
  {
    step: '01',
    title: 'Drop your video',
    description: 'Paste a YouTube link or upload a video file. Supports MP4, MOV, and AVI up to 500MB.',
  },
  {
    step: '02',
    title: 'AI does the work',
    description: 'ClippedAI transcribes, finds the best moments, reframes for 9:16, and adds subtitles — all automatically.',
  },
  {
    step: '03',
    title: 'Download & post',
    description: 'Get your clips in minutes. Download them individually or all at once, ready for TikTok, Reels, and Shorts.',
  },
];

// ===== FAQ =====
export const FAQ_ITEMS = [
  {
    question: 'How does ClippedAI work?',
    answer: 'ClippedAI uses AI to transcribe your video via AssemblyAI, then uses Google Gemini to analyze the transcript and identify the most engaging moments. It then extracts those segments, applies face-tracked vertical reframing using OpenCV, adds word-synced subtitles, and delivers finished clips ready for social media.',
  },
  {
    question: 'What types of videos work best?',
    answer: 'ClippedAI works with any video that has spoken content — podcasts, interviews, vlogs, lectures, webinars, gaming commentary, and more. It performs best with clear audio and at least one visible speaker.',
  },
  {
    question: 'What languages are supported?',
    answer: 'ClippedAI currently supports English audio for transcription and clip generation. We\'re working on expanding language support in future updates.',
  },
  {
    question: 'Can I customize the subtitles?',
    answer: 'Yes! You can choose from multiple caption styles, customize fonts (including uploading your own), adjust colors, and control positioning. Subtitles are animated word-by-word for maximum engagement.',
  },
  {
    question: 'Is ClippedAI free?',
    answer: 'ClippedAI offers a free tier to get started. Upload a video and generate your first clips at no cost. For higher volumes, we offer affordable credit packs.',
  },
  {
    question: 'How long does processing take?',
    answer: 'Processing time depends on video length. A 10-minute video typically takes 3-5 minutes. Longer videos may take proportionally longer. You can close the tab and come back — your clips will be ready.',
  },
];

// ===== Footer =====
export const FOOTER_LINKS = {
  product: {
    title: 'Product',
    links: [
      { label: 'Features', href: '#features' },
      { label: 'How it works', href: '#how-it-works' },
      { label: 'FAQ', href: '#faq' },
    ],
  },
  resources: {
    title: 'Resources',
    links: [
      { label: 'Help Center', href: '#' },
      { label: 'Blog', href: '#' },
    ],
  },
  legal: {
    title: 'Legal',
    links: [
      { label: 'Terms of Service', href: '#' },
      { label: 'Privacy Policy', href: '#' },
    ],
  },
};

export const SOCIAL_LINKS = [
  { name: 'Twitter', href: '#' },
  { name: 'Instagram', href: '#' },
];
