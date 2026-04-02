-- Remove unused Post model (T3 scaffolding)
DROP TABLE IF EXISTS "Post";

-- Remove unused VerificationToken model
DROP TABLE IF EXISTS "VerificationToken";

-- Remove posts relation from User (column doesn't exist since it's a relation, not a column)

-- Add performance indexes
CREATE INDEX IF NOT EXISTS "UploadedFile_userId_idx" ON "UploadedFile"("userId");
CREATE INDEX IF NOT EXISTS "Clip_userId_idx" ON "Clip"("userId");
CREATE INDEX IF NOT EXISTS "Clip_uploadedFileId_idx" ON "Clip"("uploadedFileId");
