import { PrismaClient } from '@prisma/client'
const prisma = new PrismaClient()

async function main() {
  const task = await prisma.uploadedFile.findUnique({
    where: { id: "cmov9av0600011pawqpi1diwz" }
  })
  console.log(task);
}

main()
