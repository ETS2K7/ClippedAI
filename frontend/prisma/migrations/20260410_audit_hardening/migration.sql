-- Migration: audit_hardening
-- Generated: 2026-04-10
--
-- Changes:
--   1. Replace single-column userId index on UploadedFile with composite (userId, createdAt)
--   2. Add VerificationToken table (required by NextAuth Prisma adapter)

-- 1. Drop the old single-column index and create composite index
--    The old index name from the init migration: UploadedFile_userId_idx
DROP INDEX IF EXISTS "UploadedFile_userId_idx";
CREATE INDEX IF NOT EXISTS "UploadedFile_userId_createdAt_idx" ON "UploadedFile" ("userId", "createdAt");

-- 2. Create VerificationToken table for NextAuth email verification support
CREATE TABLE IF NOT EXISTS "VerificationToken" (
    "identifier" TEXT NOT NULL,
    "token"      TEXT NOT NULL,
    "expires"    TIMESTAMP(3) NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS "VerificationToken_token_key"
    ON "VerificationToken" ("token");

CREATE UNIQUE INDEX IF NOT EXISTS "VerificationToken_identifier_token_key"
    ON "VerificationToken" ("identifier", "token");
