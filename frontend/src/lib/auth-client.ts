import { signIn, signOut, useSession as nextAuthUseSession } from "next-auth/react";

export function useSession() {
  try {
    const session = nextAuthUseSession();
    return { data: session?.data || null, isPending: session?.status === "loading" };
  } catch {
    return { data: null, isPending: false };
  }
}

export { signIn, signOut };
