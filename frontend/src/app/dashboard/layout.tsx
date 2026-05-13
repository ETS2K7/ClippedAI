"use server";

import { redirect } from "next/navigation";
import type { ReactNode } from "react";
import { Toaster } from "~/components/ui/sonner";
import { auth } from "~/server/auth";
import { SessionProvider } from "~/components/providers/session-provider";

export default async function DashboardLayout({
  children,
}: {
  children: ReactNode;
}) {
  const session = await auth();

  if (!session?.user?.id) {
    redirect("/auth/oauth/login");
  }

  return (
    <SessionProvider session={session}>
      <div className="flex min-h-screen flex-col bg-[#0a0a0f]">
        <main className="w-full flex-1">{children}</main>
        <Toaster />
      </div>
    </SessionProvider>
  );
}
