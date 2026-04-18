import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function main() {
  const totalClips = await prisma.clip.count();
  console.log(`Total clips in database: ${totalClips}`);

  if (totalClips === 0) {
    console.log('No clips found in database');
    return;
  }

  const clips = await prisma.clip.findMany({
    orderBy: { createdAt: 'desc' },
    take: 5,
    select: {
      id: true,
      s3Key: true,
      thumbnailKeys: true,
      createdAt: true,
    },
  });

  console.log('Recent clips:');
  console.table(clips);
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
