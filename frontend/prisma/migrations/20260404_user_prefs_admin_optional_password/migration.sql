-- Migration: add_user_prefs_admin_optional_password
-- Generated: 2026-04-04
--
-- Changes:
--   1. password column: NOT NULL → NULL  (enables future OAuth providers)
--   2. isAdmin column: new BOOLEAN NOT NULL DEFAULT false
--   3. prefFontFamily, prefFontSize, prefFontColor: user caption preferences

-- 1. Make password nullable
ALTER TABLE "User" ALTER COLUMN "password" DROP NOT NULL;

-- 2. Add isAdmin flag
ALTER TABLE "User" ADD COLUMN IF NOT EXISTS "isAdmin" BOOLEAN NOT NULL DEFAULT false;

-- 3. Add caption preference columns with sensible defaults
ALTER TABLE "User" ADD COLUMN IF NOT EXISTS "prefFontFamily" TEXT NOT NULL DEFAULT 'TikTokSans-Regular';
ALTER TABLE "User" ADD COLUMN IF NOT EXISTS "prefFontSize"   INTEGER NOT NULL DEFAULT 24;
ALTER TABLE "User" ADD COLUMN IF NOT EXISTS "prefFontColor"  TEXT NOT NULL DEFAULT '#FFFFFF';
