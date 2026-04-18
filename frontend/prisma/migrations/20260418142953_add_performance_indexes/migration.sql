/*
  Warnings:

  - You are about to drop the column `hlsKey` on the `Clip` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "Clip" DROP COLUMN "hlsKey";

-- CreateIndex
CREATE INDEX "Clip_createdAt_idx" ON "Clip"("createdAt");

-- CreateIndex
CREATE INDEX "UploadedFile_status_idx" ON "UploadedFile"("status");
