"use server";

import { redirect } from "next/navigation";
import type { ReactNode } from "react";
import { auth } from "~/server/auth";
import { SessionProvider } from "~/components/providers/session-provider";

export default async function SettingsLayout({
  children,
}: {
  children: ReactNode;
}) {
  const session = await auth();
  if (!session?.user?.id) {
    redirect("/auth/oauth/login");
  }
  return <>{children}</>;
}
