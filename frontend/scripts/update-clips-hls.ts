import { PrismaClient } from '@prisma/client';
import { S3Client, ListObjectsV2Command } from '@aws-sdk/client-s3';

const prisma = new PrismaClient();
const s3 = new S3Client({
  region: process.env.AWS_REGION || 'ap-south-1',
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
  },
});

async function main() {
  const bucket = process.env.S3_BUCKET_NAME || 'clippedai-ap-south-1';

  // Get all clips without hlsKey
  const clipsWithoutHls = await prisma.clip.findMany({
    where: { hlsKey: null } as any,
    select: { id: true, s3Key: true },
  });

  console.log(`Found ${clipsWithoutHls.length} clips without hlsKey`);

  for (const clip of clipsWithoutHls) {
    // Extract the base path from s3Key
    const basePath = clip.s3Key.substring(0, clip.s3Key.lastIndexOf('/'));
    
    // Check if HLS segments exist for this clip
    const hlsKey = `${basePath}/hls_2/master.m3u8`;
    
    try {
      await s3.send(
        new ListObjectsV2Command({
          Bucket: bucket,
          Prefix: `${basePath}/hls_2/`,
          MaxKeys: 1,
        })
      );
      
      // If HLS segments exist, update the clip
      await prisma.clip.update({
        where: { id: clip.id },
        data: { hlsKey } as any,
      });
      
      console.log(`✓ Updated clip ${clip.id} with hlsKey: ${hlsKey}`);
    } catch (error) {
      console.log(`✗ No HLS segments found for clip ${clip.id}`);
    }
  }

  console.log('Done!');
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
