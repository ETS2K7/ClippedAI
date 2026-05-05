-- Migrate all users still on the old TikTokSans-Regular default to Komika Axis.
-- This is a data-only migration — no schema structure change.
UPDATE "User"
SET "prefFontFamily" = 'Komika Axis'
WHERE "prefFontFamily" = 'TikTokSans-Regular';
