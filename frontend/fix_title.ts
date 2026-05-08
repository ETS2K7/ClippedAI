import { PrismaClient } from '@prisma/client'
const prisma = new PrismaClient()

async function main() {
  await prisma.uploadedFile.update({
    where: { id: "cmov9av0600011pawqpi1diwz" },
    data: { displayName: "Millie Bobby Brown & Chris Pratt Argue Over The Biggest Debates | Agree To Disagree" }
  })
  console.log("Title updated manually");
}

main()
