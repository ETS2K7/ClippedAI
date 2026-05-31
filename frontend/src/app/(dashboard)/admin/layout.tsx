import { redirect } from "next/navigation";
import { requireAdmin } from "~/lib/require-admin";
import { SessionProvider } from "~/components/providers/session-provider";

export default async function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const result = await requireAdmin();

  if ("error" in result || result instanceof Response) {
    redirect("/dashboard");
  }

  // At this point we are guaranteed to have a session
  // since requireAdmin only returns session property if authorized.
  const { session } = result as {
    session: import("next-auth").Session & {
      user: { id: string; isAdmin: boolean };
    };
  };

  return <SessionProvider session={session}>{children}</SessionProvider>;
}
