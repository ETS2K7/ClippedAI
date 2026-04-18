-- CreateIndex
CREATE INDEX "Clip_uploadedFileId_createdAt_idx" ON "Clip"("uploadedFileId", "createdAt");

-- CreateIndex
CREATE INDEX "Clip_userId_createdAt_idx" ON "Clip"("userId", "createdAt");

-- CreateIndex
CREATE INDEX "UploadedFile_userId_status_idx" ON "UploadedFile"("userId", "status");
