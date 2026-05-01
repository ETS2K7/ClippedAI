import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

async function main() {
  const oldAdminEmail = 'admin@clippedai.app';
  const newAdminEmail = 'ebelthomasseiko@gmail.com';

  const oldAdmin = await prisma.user.findUnique({ where: { email: oldAdminEmail } });
  const newAdmin = await prisma.user.findUnique({ where: { email: newAdminEmail } });

  if (!oldAdmin) {
    console.log("Old admin not found. Migration aborted.");
    return;
  }
  if (!newAdmin) {
    console.log("New admin not found. Migration aborted.");
    return;
  }

  // 1. Assign account creation date and promote to admin
  await prisma.user.update({
    where: { email: newAdminEmail },
    data: {
      isAdmin: true,
      createdAt: oldAdmin.createdAt,
    }
  });

  // 2. Re-assign raw files and clips (prevents Cascade deletion of user data)
  await prisma.uploadedFile.updateMany({
    where: { userId: oldAdmin.id },
    data: { userId: newAdmin.id }
  });

  await prisma.clip.updateMany({
    where: { userId: oldAdmin.id },
    data: { userId: newAdmin.id }
  });

  // 3. Delete old admin account
  await prisma.user.delete({
    where: { email: oldAdminEmail }
  });

  console.log("Migration complete! Transferred data and deleted old admin.");
}

main()
  .then(async () => {
    await prisma.$disconnect()
  })
  .catch(async (e) => {
    console.error(e)
    await prisma.$disconnect()
    process.exit(1)
  })
