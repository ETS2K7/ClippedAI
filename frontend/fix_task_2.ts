const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

async function main() {
  const taskId = 'cmowwdy8y0001wymcnos8x6o0';
  const task = await prisma.uploadedFile.findUnique({ where: { id: taskId } });
  
  if (!task) {
    console.log("Task not found");
    return;
  }

  const duration = (new Date(task.updatedAt).getTime() - new Date(task.createdAt).getTime()) / 1000;
  
  await prisma.uploadedFile.update({
    where: { id: taskId },
    data: { 
      displayName: "Habit Stacking: Structure Your Day for Peak Focus | James Clear & Dr. Andrew Huberman",
      processingTime: duration
    }
  });
  console.log(`Updated task ${taskId}. New duration: ${duration}s`);
}

main().finally(() => prisma.$disconnect());
