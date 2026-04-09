# Details

Date : 2026-04-09 16:28:27

Directory /Users/ebelthomasseiko/ClippedAI

Total : 165 files,  24621 codes, 598 comments, 1984 blanks, all 27203 lines

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [.github/workflows/deploy.yml](/.github/workflows/deploy.yml) | YAML | 135 | 7 | 20 | 162 |
| [DEPLOY.md](/DEPLOY.md) | Markdown | 136 | 0 | 65 | 201 |
| [README.md](/README.md) | Markdown | 107 | 0 | 24 | 131 |
| [backend/config.py](/backend/config.py) | Python | 48 | 6 | 17 | 71 |
| [backend/main.py](/backend/main.py) | Python | 188 | 6 | 40 | 234 |
| [backend/modal_fast_asd.py](/backend/modal_fast_asd.py) | Python | 62 | 0 | 9 | 71 |
| [backend/read_secrets.py](/backend/read_secrets.py) | Python | 23 | 0 | 4 | 27 |
| [backend/requirements.txt](/backend/requirements.txt) | pip requirements | 13 | 0 | 0 | 13 |
| [backend/run_local.py](/backend/run_local.py) | Python | 48 | 10 | 18 | 76 |
| [backend/src/downloader.py](/backend/src/downloader.py) | Python | 64 | 1 | 9 | 74 |
| [backend/src/llm.py](/backend/src/llm.py) | Python | 106 | 5 | 25 | 136 |
| [backend/src/signal_helpers.py](/backend/src/signal_helpers.py) | Python | 63 | 0 | 7 | 70 |
| [backend/src/subtitles.py](/backend/src/subtitles.py) | Python | 108 | 3 | 27 | 138 |
| [backend/src/transcriber.py](/backend/src/transcriber.py) | Python | 60 | 2 | 17 | 79 |
| [backend/src/video_processing.py](/backend/src/video_processing.py) | Python | 497 | 85 | 100 | 682 |
| [backend/tests/conftest.py](/backend/tests/conftest.py) | Python | 3 | 1 | 2 | 6 |
| [backend/tests/test_video_processing.py](/backend/tests/test_video_processing.py) | Python | 48 | 7 | 29 | 84 |
| [deploy/bootstrap.sh](/deploy/bootstrap.sh) | Shell Script | 102 | 13 | 22 | 137 |
| [deploy/github-secrets.md](/deploy/github-secrets.md) | Markdown | 51 | 0 | 25 | 76 |
| [deploy/oci-premium.sh](/deploy/oci-premium.sh) | Shell Script | 60 | 6 | 16 | 82 |
| [deploy/oci-provision.sh](/deploy/oci-provision.sh) | Shell Script | 149 | 22 | 21 | 192 |
| [deploy/oci-retry.sh](/deploy/oci-retry.sh) | Shell Script | 67 | 13 | 17 | 97 |
| [deploy/setup-ssl.sh](/deploy/setup-ssl.sh) | Shell Script | 65 | 8 | 15 | 88 |
| [docker-compose.yml](/docker-compose.yml) | YAML | 44 | 0 | 3 | 47 |
| [frontend/Dockerfile](/frontend/Dockerfile) | Docker | 33 | 8 | 13 | 54 |
| [frontend/README.md](/frontend/README.md) | Markdown | 18 | 0 | 12 | 30 |
| [frontend/components.json](/frontend/components.json) | JSON | 21 | 0 | 0 | 21 |
| [frontend/eslint.config.js](/frontend/eslint.config.js) | JavaScript | 57 | 3 | 3 | 63 |
| [frontend/next.config.js](/frontend/next.config.js) | JavaScript | 58 | 2 | 8 | 68 |
| [frontend/package-lock.json](/frontend/package-lock.json) | JSON | 10,940 | 0 | 1 | 10,941 |
| [frontend/package.json](/frontend/package.json) | JSON | 84 | 0 | 1 | 85 |
| [frontend/postcss.config.js](/frontend/postcss.config.js) | JavaScript | 5 | 0 | 1 | 6 |
| [frontend/prettier.config.js](/frontend/prettier.config.js) | JavaScript | 4 | 0 | 1 | 5 |
| [frontend/prisma/migrations/0001_init.sql](/frontend/prisma/migrations/0001_init.sql) | MS SQL | 80 | 21 | 28 | 129 |
| [frontend/prisma/migrations/20260101000000_init/migration.sql](/frontend/prisma/migrations/20260101000000_init/migration.sql) | MS SQL | 80 | 21 | 28 | 129 |
| [frontend/prisma/migrations/20260402_cleanup_remove_post_add_indexes/migration.sql](/frontend/prisma/migrations/20260402_cleanup_remove_post_add_indexes/migration.sql) | MS SQL | 5 | 4 | 4 | 13 |
| [frontend/prisma/migrations/20260404_user_prefs_admin_optional_password/migration.sql](/frontend/prisma/migrations/20260404_user_prefs_admin_optional_password/migration.sql) | MS SQL | 5 | 10 | 4 | 19 |
| [frontend/public/icons/bg-line-2.svg](/frontend/public/icons/bg-line-2.svg) | XML | 1 | 0 | 0 | 1 |
| [frontend/public/icons/bg-line.svg](/frontend/public/icons/bg-line.svg) | XML | 1 | 0 | 0 | 1 |
| [frontend/public/icons/nav-arrow-down.svg](/frontend/public/icons/nav-arrow-down.svg) | XML | 1 | 0 | 0 | 1 |
| [frontend/public/icons/search-header-bg.svg](/frontend/public/icons/search-header-bg.svg) | XML | 1 | 0 | 0 | 1 |
| [frontend/public/icons/xmark.svg](/frontend/public/icons/xmark.svg) | XML | 1 | 0 | 0 | 1 |
| [frontend/src/actions/auth.ts](/frontend/src/actions/auth.ts) | TypeScript | 37 | 0 | 10 | 47 |
| [frontend/src/actions/generation.ts](/frontend/src/actions/generation.ts) | TypeScript | 33 | 0 | 7 | 40 |
| [frontend/src/actions/s3.ts](/frontend/src/actions/s3.ts) | TypeScript | 46 | 0 | 9 | 55 |
| [frontend/src/app/api/admin/users/[userId]/admin/route.ts](/frontend/src/app/api/admin/users/%5BuserId%5D/admin/route.ts) | TypeScript | 57 | 1 | 11 | 69 |
| [frontend/src/app/api/auth/[...nextauth]/route.ts](/frontend/src/app/api/auth/%5B...nextauth%5D/route.ts) | TypeScript | 2 | 0 | 2 | 4 |
| [frontend/src/app/api/broll/status/route.ts](/frontend/src/app/api/broll/status/route.ts) | TypeScript | 4 | 0 | 2 | 6 |
| [frontend/src/app/api/caption-templates/route.ts](/frontend/src/app/api/caption-templates/route.ts) | TypeScript | 123 | 1 | 4 | 128 |
| [frontend/src/app/api/feedback/route.ts](/frontend/src/app/api/feedback/route.ts) | TypeScript | 37 | 2 | 9 | 48 |
| [frontend/src/app/api/fonts/[name]/route.ts](/frontend/src/app/api/fonts/%5Bname%5D/route.ts) | TypeScript | 78 | 8 | 11 | 97 |
| [frontend/src/app/api/fonts/route.ts](/frontend/src/app/api/fonts/route.ts) | TypeScript | 32 | 1 | 4 | 37 |
| [frontend/src/app/api/health/route.ts](/frontend/src/app/api/health/route.ts) | TypeScript | 19 | 2 | 5 | 26 |
| [frontend/src/app/api/preferences/route.ts](/frontend/src/app/api/preferences/route.ts) | TypeScript | 63 | 1 | 14 | 78 |
| [frontend/src/app/api/tasks/[id]/progress/route.ts](/frontend/src/app/api/tasks/%5Bid%5D/progress/route.ts) | TypeScript | 105 | 7 | 16 | 128 |
| [frontend/src/app/api/tasks/[id]/route.ts](/frontend/src/app/api/tasks/%5Bid%5D/route.ts) | TypeScript | 126 | 7 | 22 | 155 |
| [frontend/src/app/api/tasks/create/route.ts](/frontend/src/app/api/tasks/create/route.ts) | TypeScript | 119 | 8 | 21 | 148 |
| [frontend/src/app/api/tasks/route.ts](/frontend/src/app/api/tasks/route.ts) | TypeScript | 46 | 0 | 8 | 54 |
| [frontend/src/app/api/upload/route.ts](/frontend/src/app/api/upload/route.ts) | TypeScript | 79 | 5 | 15 | 99 |
| [frontend/src/app/api/webhooks/modal/route.ts](/frontend/src/app/api/webhooks/modal/route.ts) | TypeScript | 76 | 7 | 14 | 97 |
| [frontend/src/app/dashboard/layout.tsx](/frontend/src/app/dashboard/layout.tsx) | TypeScript JSX | 24 | 0 | 5 | 29 |
| [frontend/src/app/dashboard/loading.tsx](/frontend/src/app/dashboard/loading.tsx) | TypeScript JSX | 9 | 0 | 2 | 11 |
| [frontend/src/app/dashboard/page.tsx](/frontend/src/app/dashboard/page.tsx) | TypeScript JSX | 1,049 | 44 | 100 | 1,193 |
| [frontend/src/app/layout.tsx](/frontend/src/app/layout.tsx) | TypeScript JSX | 35 | 0 | 8 | 43 |
| [frontend/src/app/list/layout.tsx](/frontend/src/app/list/layout.tsx) | TypeScript JSX | 12 | 0 | 3 | 15 |
| [frontend/src/app/list/page.tsx](/frontend/src/app/list/page.tsx) | TypeScript JSX | 682 | 13 | 58 | 753 |
| [frontend/src/app/login/page.tsx](/frontend/src/app/login/page.tsx) | TypeScript JSX | 40 | 4 | 10 | 54 |
| [frontend/src/app/not-found.tsx](/frontend/src/app/not-found.tsx) | TypeScript JSX | 25 | 1 | 3 | 29 |
| [frontend/src/app/page.tsx](/frontend/src/app/page.tsx) | TypeScript JSX | 31 | 0 | 3 | 34 |
| [frontend/src/app/settings/layout.tsx](/frontend/src/app/settings/layout.tsx) | TypeScript JSX | 12 | 0 | 3 | 15 |
| [frontend/src/app/settings/page.tsx](/frontend/src/app/settings/page.tsx) | TypeScript JSX | 324 | 15 | 34 | 373 |
| [frontend/src/app/signup/page.tsx](/frontend/src/app/signup/page.tsx) | TypeScript JSX | 40 | 4 | 10 | 54 |
| [frontend/src/app/tasks/[id]/edit/page.tsx](/frontend/src/app/tasks/%5Bid%5D/edit/page.tsx) | TypeScript JSX | 851 | 0 | 98 | 949 |
| [frontend/src/app/tasks/[id]/page.tsx](/frontend/src/app/tasks/%5Bid%5D/page.tsx) | TypeScript JSX | 428 | 8 | 33 | 469 |
| [frontend/src/app/tasks/layout.tsx](/frontend/src/app/tasks/layout.tsx) | TypeScript JSX | 12 | 0 | 3 | 15 |
| [frontend/src/components/admin/admin-user-toggle.tsx](/frontend/src/components/admin/admin-user-toggle.tsx) | TypeScript JSX | 52 | 0 | 10 | 62 |
| [frontend/src/components/app-shell.tsx](/frontend/src/components/app-shell.tsx) | TypeScript JSX | 195 | 11 | 17 | 223 |
| [frontend/src/components/clip-display.tsx](/frontend/src/components/clip-display.tsx) | TypeScript JSX | 81 | 0 | 8 | 89 |
| [frontend/src/components/datafast-identity.tsx](/frontend/src/components/datafast-identity.tsx) | TypeScript JSX | 28 | 0 | 10 | 38 |
| [frontend/src/components/dynamic-video-player.tsx](/frontend/src/components/dynamic-video-player.tsx) | TypeScript JSX | 42 | 0 | 5 | 47 |
| [frontend/src/components/feedback-button.tsx](/frontend/src/components/feedback-button.tsx) | TypeScript JSX | 112 | 0 | 14 | 126 |
| [frontend/src/components/landing-v2/bento-grid.tsx](/frontend/src/components/landing-v2/bento-grid.tsx) | TypeScript JSX | 95 | 0 | 13 | 108 |
| [frontend/src/components/landing-v2/flip-words.tsx](/frontend/src/components/landing-v2/flip-words.tsx) | TypeScript JSX | 79 | 0 | 5 | 84 |
| [frontend/src/components/landing-v2/floating-nav.tsx](/frontend/src/components/landing-v2/floating-nav.tsx) | TypeScript JSX | 66 | 3 | 4 | 73 |
| [frontend/src/components/landing-v2/grid-background.tsx](/frontend/src/components/landing-v2/grid-background.tsx) | TypeScript JSX | 12 | 0 | 2 | 14 |
| [frontend/src/components/landing-v2/infinite-marquee.tsx](/frontend/src/components/landing-v2/infinite-marquee.tsx) | TypeScript JSX | 106 | 0 | 10 | 116 |
| [frontend/src/components/landing-v2/moving-border-button.tsx](/frontend/src/components/landing-v2/moving-border-button.tsx) | TypeScript JSX | 34 | 0 | 2 | 36 |
| [frontend/src/components/landing-v2/sections/cta-section.tsx](/frontend/src/components/landing-v2/sections/cta-section.tsx) | TypeScript JSX | 28 | 1 | 6 | 35 |
| [frontend/src/components/landing-v2/sections/features-section.tsx](/frontend/src/components/landing-v2/sections/features-section.tsx) | TypeScript JSX | 100 | 0 | 5 | 105 |
| [frontend/src/components/landing-v2/sections/footer-section.tsx](/frontend/src/components/landing-v2/sections/footer-section.tsx) | TypeScript JSX | 64 | 0 | 6 | 70 |
| [frontend/src/components/landing-v2/sections/hero-section.tsx](/frontend/src/components/landing-v2/sections/hero-section.tsx) | TypeScript JSX | 42 | 1 | 8 | 51 |
| [frontend/src/components/landing-v2/sections/how-it-works.tsx](/frontend/src/components/landing-v2/sections/how-it-works.tsx) | TypeScript JSX | 68 | 2 | 9 | 79 |
| [frontend/src/components/landing-v2/sections/social-proof.tsx](/frontend/src/components/landing-v2/sections/social-proof.tsx) | TypeScript JSX | 24 | 0 | 3 | 27 |
| [frontend/src/components/landing-v2/sections/stats-section.tsx](/frontend/src/components/landing-v2/sections/stats-section.tsx) | TypeScript JSX | 28 | 0 | 3 | 31 |
| [frontend/src/components/landing-v2/spotlight.tsx](/frontend/src/components/landing-v2/spotlight.tsx) | TypeScript JSX | 42 | 0 | 2 | 44 |
| [frontend/src/components/landing-v2/text-generate.tsx](/frontend/src/components/landing-v2/text-generate.tsx) | TypeScript JSX | 55 | 0 | 5 | 60 |
| [frontend/src/components/landing-v3/interactive-hero.tsx](/frontend/src/components/landing-v3/interactive-hero.tsx) | TypeScript JSX | 469 | 14 | 39 | 522 |
| [frontend/src/components/landing-v3/kinetic-typography.tsx](/frontend/src/components/landing-v3/kinetic-typography.tsx) | TypeScript JSX | 47 | 3 | 13 | 63 |
| [frontend/src/components/landing-v3/sticky-narrative.tsx](/frontend/src/components/landing-v3/sticky-narrative.tsx) | TypeScript JSX | 166 | 23 | 26 | 215 |
| [frontend/src/components/landing-v3/void-cta.tsx](/frontend/src/components/landing-v3/void-cta.tsx) | TypeScript JSX | 64 | 3 | 13 | 80 |
| [frontend/src/components/landing/animations/FadeIn.tsx](/frontend/src/components/landing/animations/FadeIn.tsx) | TypeScript JSX | 30 | 0 | 4 | 34 |
| [frontend/src/components/landing/animations/ScrollReveal.tsx](/frontend/src/components/landing/animations/ScrollReveal.tsx) | TypeScript JSX | 50 | 0 | 5 | 55 |
| [frontend/src/components/landing/animations/SharedLayoutProvider.tsx](/frontend/src/components/landing/animations/SharedLayoutProvider.tsx) | TypeScript JSX | 6 | 0 | 3 | 9 |
| [frontend/src/components/landing/layout/CookieBanner.tsx](/frontend/src/components/landing/layout/CookieBanner.tsx) | TypeScript JSX | 60 | 0 | 8 | 68 |
| [frontend/src/components/landing/layout/Footer.tsx](/frontend/src/components/landing/layout/Footer.tsx) | TypeScript JSX | 90 | 3 | 9 | 102 |
| [frontend/src/components/landing/layout/Navbar.tsx](/frontend/src/components/landing/layout/Navbar.tsx) | TypeScript JSX | 227 | 6 | 18 | 251 |
| [frontend/src/components/landing/sections/AIEditor.tsx](/frontend/src/components/landing/sections/AIEditor.tsx) | TypeScript JSX | 82 | 3 | 8 | 93 |
| [frontend/src/components/landing/sections/Autopilot.tsx](/frontend/src/components/landing/sections/Autopilot.tsx) | TypeScript JSX | 47 | 5 | 6 | 58 |
| [frontend/src/components/landing/sections/CTASection.tsx](/frontend/src/components/landing/sections/CTASection.tsx) | TypeScript JSX | 76 | 2 | 7 | 85 |
| [frontend/src/components/landing/sections/ClipAnything.tsx](/frontend/src/components/landing/sections/ClipAnything.tsx) | TypeScript JSX | 66 | 7 | 8 | 81 |
| [frontend/src/components/landing/sections/FAQ.tsx](/frontend/src/components/landing/sections/FAQ.tsx) | TypeScript JSX | 81 | 2 | 6 | 89 |
| [frontend/src/components/landing/sections/Hero.tsx](/frontend/src/components/landing/sections/Hero.tsx) | TypeScript JSX | 154 | 13 | 15 | 182 |
| [frontend/src/components/landing/sections/ScaleSection.tsx](/frontend/src/components/landing/sections/ScaleSection.tsx) | TypeScript JSX | 44 | 0 | 5 | 49 |
| [frontend/src/components/landing/ui/AuthModal.tsx](/frontend/src/components/landing/ui/AuthModal.tsx) | TypeScript JSX | 106 | 9 | 16 | 131 |
| [frontend/src/components/landing/ui/FloatingCTA.tsx](/frontend/src/components/landing/ui/FloatingCTA.tsx) | TypeScript JSX | 51 | 0 | 8 | 59 |
| [frontend/src/components/landing/ui/HeroCarousel.tsx](/frontend/src/components/landing/ui/HeroCarousel.tsx) | TypeScript JSX | 163 | 13 | 18 | 194 |
| [frontend/src/components/login-form.tsx](/frontend/src/components/login-form.tsx) | TypeScript JSX | 128 | 0 | 10 | 138 |
| [frontend/src/components/providers/session-provider.tsx](/frontend/src/components/providers/session-provider.tsx) | TypeScript JSX | 5 | 0 | 2 | 7 |
| [frontend/src/components/signup-form.tsx](/frontend/src/components/signup-form.tsx) | TypeScript JSX | 133 | 0 | 11 | 144 |
| [frontend/src/components/ui/alert-dialog.tsx](/frontend/src/components/ui/alert-dialog.tsx) | TypeScript JSX | 127 | 0 | 15 | 142 |
| [frontend/src/components/ui/alert.tsx](/frontend/src/components/ui/alert.tsx) | TypeScript JSX | 60 | 0 | 7 | 67 |
| [frontend/src/components/ui/avatar.tsx](/frontend/src/components/ui/avatar.tsx) | TypeScript JSX | 47 | 0 | 7 | 54 |
| [frontend/src/components/ui/badge.tsx](/frontend/src/components/ui/badge.tsx) | TypeScript JSX | 41 | 0 | 6 | 47 |
| [frontend/src/components/ui/button.tsx](/frontend/src/components/ui/button.tsx) | TypeScript JSX | 55 | 0 | 6 | 61 |
| [frontend/src/components/ui/card.tsx](/frontend/src/components/ui/card.tsx) | TypeScript JSX | 47 | 0 | 10 | 57 |
| [frontend/src/components/ui/checkbox.tsx](/frontend/src/components/ui/checkbox.tsx) | TypeScript JSX | 28 | 0 | 5 | 33 |
| [frontend/src/components/ui/dropdown-menu.tsx](/frontend/src/components/ui/dropdown-menu.tsx) | TypeScript JSX | 239 | 0 | 19 | 258 |
| [frontend/src/components/ui/input.tsx](/frontend/src/components/ui/input.tsx) | TypeScript JSX | 18 | 0 | 4 | 22 |
| [frontend/src/components/ui/label.tsx](/frontend/src/components/ui/label.tsx) | TypeScript JSX | 20 | 0 | 5 | 25 |
| [frontend/src/components/ui/popover.tsx](/frontend/src/components/ui/popover.tsx) | TypeScript JSX | 79 | 0 | 11 | 90 |
| [frontend/src/components/ui/progress.tsx](/frontend/src/components/ui/progress.tsx) | TypeScript JSX | 27 | 0 | 5 | 32 |
| [frontend/src/components/ui/select.tsx](/frontend/src/components/ui/select.tsx) | TypeScript JSX | 172 | 0 | 14 | 186 |
| [frontend/src/components/ui/separator.tsx](/frontend/src/components/ui/separator.tsx) | TypeScript JSX | 24 | 0 | 5 | 29 |
| [frontend/src/components/ui/sheet.tsx](/frontend/src/components/ui/sheet.tsx) | TypeScript JSX | 130 | 0 | 14 | 144 |
| [frontend/src/components/ui/skeleton.tsx](/frontend/src/components/ui/skeleton.tsx) | TypeScript JSX | 11 | 0 | 3 | 14 |
| [frontend/src/components/ui/slider.tsx](/frontend/src/components/ui/slider.tsx) | TypeScript JSX | 24 | 0 | 5 | 29 |
| [frontend/src/components/ui/sonner.tsx](/frontend/src/components/ui/sonner.tsx) | TypeScript JSX | 21 | 0 | 5 | 26 |
| [frontend/src/components/ui/switch.tsx](/frontend/src/components/ui/switch.tsx) | TypeScript JSX | 27 | 0 | 5 | 32 |
| [frontend/src/components/ui/table.tsx](/frontend/src/components/ui/table.tsx) | TypeScript JSX | 105 | 0 | 12 | 117 |
| [frontend/src/components/ui/tabs.tsx](/frontend/src/components/ui/tabs.tsx) | TypeScript JSX | 59 | 0 | 8 | 67 |
| [frontend/src/components/ui/textarea.tsx](/frontend/src/components/ui/textarea.tsx) | TypeScript JSX | 15 | 0 | 4 | 19 |
| [frontend/src/components/ui/tooltip.tsx](/frontend/src/components/ui/tooltip.tsx) | TypeScript JSX | 50 | 0 | 8 | 58 |
| [frontend/src/env.js](/frontend/src/env.js) | JavaScript | 49 | 3 | 4 | 56 |
| [frontend/src/lib/api-error.ts](/frontend/src/lib/api-error.ts) | TypeScript | 55 | 1 | 13 | 69 |
| [frontend/src/lib/app-flags.ts](/frontend/src/lib/app-flags.ts) | TypeScript | 2 | 0 | 1 | 3 |
| [frontend/src/lib/auth-client.ts](/frontend/src/lib/auth-client.ts) | TypeScript | 13 | 0 | 3 | 16 |
| [frontend/src/lib/auth.ts](/frontend/src/lib/auth.ts) | TypeScript | 7 | 0 | 3 | 10 |
| [frontend/src/lib/constants.ts](/frontend/src/lib/constants.ts) | TypeScript | 13 | 0 | 2 | 15 |
| [frontend/src/lib/datafast.ts](/frontend/src/lib/datafast.ts) | TypeScript | 81 | 0 | 25 | 106 |
| [frontend/src/lib/landing/constants.ts](/frontend/src/lib/landing/constants.ts) | TypeScript | 112 | 6 | 8 | 126 |
| [frontend/src/lib/monetization.ts](/frontend/src/lib/monetization.ts) | TypeScript | 4 | 0 | 2 | 6 |
| [frontend/src/lib/require-admin.ts](/frontend/src/lib/require-admin.ts) | TypeScript | 29 | 0 | 7 | 36 |
| [frontend/src/lib/utils.ts](/frontend/src/lib/utils.ts) | TypeScript | 5 | 0 | 2 | 7 |
| [frontend/src/schemas/auth.ts](/frontend/src/schemas/auth.ts) | TypeScript | 11 | 0 | 4 | 15 |
| [frontend/src/server/auth/config.ts](/frontend/src/server/auth/config.ts) | TypeScript | 99 | 9 | 13 | 121 |
| [frontend/src/server/auth/index.ts](/frontend/src/server/auth/index.ts) | TypeScript | 6 | 0 | 5 | 11 |
| [frontend/src/server/db.ts](/frontend/src/server/db.ts) | TypeScript | 12 | 0 | 6 | 18 |
| [frontend/src/server/s3.ts](/frontend/src/server/s3.ts) | TypeScript | 22 | 0 | 5 | 27 |
| [frontend/src/styles/globals.css](/frontend/src/styles/globals.css) | CSS | 265 | 0 | 25 | 290 |
| [frontend/tsconfig.json](/frontend/tsconfig.json) | JSON with Comments | 41 | 0 | 4 | 45 |
| [nginx/clippedai.conf](/nginx/clippedai.conf) | Properties | 68 | 15 | 14 | 97 |
| [nginx/conf.d/rate-limits.conf](/nginx/conf.d/rate-limits.conf) | Properties | 3 | 5 | 1 | 9 |
| [scripts/oci-free-hunter.sh](/scripts/oci-free-hunter.sh) | Shell Script | 212 | 30 | 40 | 282 |
| [scripts/sync-clips-from-s3.ts](/scripts/sync-clips-from-s3.ts) | TypeScript | 81 | 2 | 15 | 98 |
| [scripts/tsconfig.json](/scripts/tsconfig.json) | JSON with Comments | 9 | 0 | 1 | 10 |

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)