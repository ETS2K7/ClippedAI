import { PrismaClient } from '@prisma/client'
const prisma = new PrismaClient()

async function main() {
  const result = await prisma.uploadedFile.updateMany({
    where: { 
      status: { in: ["processing", "uploading", "queued"] },
      createdAt: { lt: new Date(Date.now() - 2 * 60 * 60 * 1000) } // older than 2 hours
    },
    data: { status: "failed" }
  })
  console.log(`Marked ${result.count} dangling tasks as failed.`);
}

main()
