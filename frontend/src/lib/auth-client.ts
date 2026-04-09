import { signIn, signOut, useSession as nextAuthUseSession } from "next-auth/react";

export function useSession() {
  try {
    const session = nextAuthUseSession();
    return { data: session?.data || null, isPending: session?.status === "loading" };
  } catch (error) {
    console.warn("[auth-client] Session hook error:", error);
    return { data: null, isPending: false };
  }
}

export { signIn, signOut };
