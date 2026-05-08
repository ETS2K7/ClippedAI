import { PrismaClient } from '@prisma/client'
const prisma = new PrismaClient()

async function main() {
  await prisma.uploadedFile.update({
    where: { id: "cmowpzc7y0001w907jsgv36z8" },
    data: { 
      displayName: "Millie Bobby Brown & Chris Pratt Argue Over The Biggest Debates | Agree To Disagree",
      processingTime: 72.4
    }
  })
  console.log("Updated task");
}

main()
