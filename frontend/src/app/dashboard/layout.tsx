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
    redirect("/login");
  }

  return (
    <SessionProvider session={session}>
      <div className="flex min-h-screen flex-col bg-stone-50">
        <main className="flex-1 w-full bg-stone-50">{children}</main>
        <Toaster />
      </div>
    </SessionProvider>
  );
}
