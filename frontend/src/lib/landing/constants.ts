// ===== Navigation Links =====
export const NAV_LINKS = {
  features: {
    label: 'Features',
    items: [
      { label: 'ClipAnything', description: 'Clip any video type with AI', href: '#clip-anything' },
      { label: 'AI B-Roll', description: 'Add AI-generated B-roll footage', href: '#slide-broll' },
      { label: 'AI Captions', description: 'Auto-generate accurate captions', href: '#slide-captions' },
      { label: 'AI Reframe', description: 'Reframe videos for any platform', href: '#slide-reframe' },
    ],
  },
  solutions: {
    label: 'Solutions',
    items: [
      { label: 'For Creators', href: '#creators' },
      { label: 'For Marketers', href: '#marketers' },
      { label: 'For Agencies', href: '#agencies' },
      { label: 'For Podcasters', href: '#podcasters' },
    ],
  },
  resources: {
    label: 'Resources',
    items: [
      { label: 'Customer stories', href: '#' },
      { label: 'Blog', href: '#' },
      { label: 'Help center', href: '#' },
      { label: 'Learning center', href: '#' },
    ],
  },
  forBusiness: {
    label: 'For business',
    items: [
      { label: 'Enterprise', href: '#' },
      { label: 'API', href: '#' },
    ],
  },
};

export const DASHBOARD_URL = '/dashboard';
export const PRICING_URL = '#pricing';
export const AGENT_OPUS_URL = '#agent-opus';

// ===== Hero Section =====
export const HERO_CONTENT = {
  badge: '#1 AI VIDEO CLIPPING TOOL',
  heading: '1 long video, 10 viral clips. Create 10x faster.',
  subheading: 'ClippedAI turns long videos into shorts, and publishes them to all social platforms in one click.',
  inputPlaceholder: 'Drop a video link',
  ctaPrimary: 'Get free clips',
  ctaSecondary: 'Upload files',
};

// ===== Trusted By Logos =====
export const TRUSTED_BY_LOGOS = [
  { name: 'NVIDIA', src: '/images/nvidia-logo.avif' },
  { name: 'GitHub', src: '/images/github-logo.avif' },
  { name: 'Visa', src: '/images/visa-logo.avif' },
  { name: 'iHeartMedia', src: '/images/iheartmedia-logo.avif' },
  { name: 'Audacy', src: '/images/audacy-logo.avif' },
  { name: 'Telefonica', src: '/images/telefonica-logo.avif' },
  { name: 'Univision', src: '/images/univision-logo.avif' },
  { name: 'ZoomInfo', src: '/images/zoominfo-logo.avif' },
  { name: 'Memphis Grizzlies', src: '/images/memphis-grizzlies-logo.avif' },
  { name: 'Chili Piper', src: '/images/chili-piper-logo.avif' },
];

// ===== Feature Sections =====
export const CLIP_ANYTHING = {
  heading: 'Turn long videos into shorts in 1 click',
  tabs: ['Vlog', 'Podcast', 'Gaming', 'Talking Head'],
  features: [
    {
      title: 'ClipAnything™',
      description: 'Every other AI clipping tool only works with video podcasts. ClipAnything is the only AI clipping model that turns any genre — vlogs, gaming, sports, interviews, explainer videos — into viral clips in 1 click.',
      videoUrl: '/videos/clip-anything.mp4'
    },
    {
      title: 'ReframeAnything™',
      description: 'The only AI reframe model that resizes any video for any platform and keeps moving subjects centered with AI object tracking. If you want more control, use manual tracking to instruct AI exactly what to follow.',
      videoUrl: '/videos/reframe.mp4'
    }
  ]
};

// ===== Scale Section =====
export const SCALE_FEATURES = [
  {
    title: 'Brand templates',
    description: 'Create consistent branded clips with custom templates, logos, and colors.',
    image: '/images/brand-templates.avif',
  },
  {
    title: 'Team workspace',
    description: 'Collaborate with your team in a shared workspace with role-based access.',
    image: '/images/team-workspace.avif',
  },
  {
    title: 'Workflow integration',
    description: 'Connect with your existing tools and automate your content pipeline.',
    image: '/images/workflow-integration.avif',
  },
];

// ===== FAQ =====
export const FAQ_ITEMS = [
  {
    question: 'How does ClippedAI work?',
    answer: 'ClippedAI leverages big data to analyze your video content in relation to the latest social and marketing trends from major platforms, and generates a comprehensive understanding of your video for a data-driven decision on content repurposing. It then picks the highlighting moments of your long video, rearranges them into a viral-worthy short and polishes it with dynamic captions, AI-relayout, smooth transition to ensure that the clip is coherent and attention-grabbing, and ends with a strong call-to-action.',
  },
  {
    question: 'What types of videos can I upload?',
    answer: 'You can use ClippedAI to clip any video type with our newest model ClipAnything. Whether it\'s talking-head videos like podcasts and interviews, vlogs, sports, TV shows, or videos with little to no dialogue, ClipAnything understands all the visual, audio and sentiment cues throughout the video, and can clip the best moments from your video. You can also use natural language prompts to clip a specific moment.',
  },
  {
    question: 'Which languages are supported?',
    answer: 'We support English, German, Spanish, French, Portuguese, Italian, Dutch, Russian, Polish, Indonesian, Ukrainian, Swedish, Turkish, Norwegian, Croatian, Romanian, Slovak, Greek, Danish, Finnish, Hungarian, Czech, Japanese, Korean, Vietnamese and more to come.',
  },
  {
    question: 'Can I add captions?',
    answer: 'Absolutely! In fact, ClippedAI automatically adds captions for you with over 97% accuracy! You can change text and edit it freely.',
  },
  {
    question: 'Is ClippedAI free to use?',
    answer: 'ClippedAI is free to use. If you are a new user, you will enjoy a 7-day free trial of our Pro Plan, which gives you 90 minutes of video processing time (~30 downloadable clips). Once your free trial ends, you can either upgrade to paid subscription, or use our free-forever plan with 60 minutes of video processing time refreshed monthly. We offer paid subscription plans with flexible upload credits, available on both a monthly and yearly basis.',
  },
  {
    question: 'I have more questions!',
    answer: 'Please join our Discord or email us at support@opus.pro if you need help or have any questions or advice for us.',
  },
];

// ===== Footer =====
export const FOOTER_LINKS = {
  company: {
    title: 'Company',
    links: [
      { label: 'About us', href: '#' },
      { label: 'Careers', href: '#', badge: "We're hiring!" },
      { label: 'Contact Us', href: '#' },
    ],
  },
  product: {
    title: 'Product',
    links: [
      { label: 'Tools', href: '#' },
      { label: 'Product changelog', href: '#' },
      { label: 'Feature Request', href: '#' },
      { label: 'ClippedAI Captions', href: '#', badge: 'Free' },
      { label: 'How does ClippedAI work', href: '#' },
    ],
  },
  resources: {
    title: 'Resources',
    links: [
      { label: 'Customer stories', href: '#' },
      { label: 'Blogs', href: '#' },
      { label: 'Become an affiliate', href: '#' },
      { label: 'ClippedAI media assets', href: '#' },
      { label: 'Help center', href: '#' },
      { label: 'Learning center', href: '#' },
    ],
  },
  bestPractices: {
    title: 'Best Practices',
    links: [
      { label: 'Hidden strategy top creators use to turn clips into 4M+ long-form views', href: '#' },
      { label: "MrBeast's Vyro: Everything You Need to Know", href: '#' },
      { label: 'From zero to monetization for a new YouTube channel in 40 days', href: '#' },
      { label: 'How to Turn Long Videos into Viral Shorts: The Ultimate Guide for 2026', href: '#' },
    ],
  },
  trustAndLegal: {
    title: 'Trust and Legal',
    links: [
      { label: 'Terms and Conditions', href: '#' },
      { label: 'Privacy Policy', href: '#' },
      { label: 'Cookies Preferences', href: '#' },
      { label: 'Accessibility Statement', href: '#' },
      { label: 'Email protection', href: '#' },
      { label: 'Trust Center', href: '#' },
    ],
  },
};

export const SOCIAL_LINKS = [
  { name: 'YouTube', href: '#' },
  { name: 'TikTok', href: '#' },
  { name: 'Twitter', href: '#' },
  { name: 'Discord', href: '#' },
  { name: 'LinkedIn', href: '#' },
  { name: 'Instagram', href: '#' },
];
