import { PrismaAdapter } from "@auth/prisma-adapter";
import { type DefaultSession, type NextAuthConfig } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import { comparePasswords } from "~/lib/auth";

import { db } from "~/server/db";

/**
 * Module augmentation for `next-auth` types. Allows us to add custom properties to the `session`
 * object and keep type safety.
 *
 * @see https://next-auth.js.org/getting-started/typescript#module-augmentation
 */
declare module "next-auth" {
  interface Session extends DefaultSession {
    user: {
      id: string;
      isAdmin: boolean;
    } & DefaultSession["user"];
  }
}


/**
 * Options for NextAuth.js used to configure adapters, providers, callbacks, etc.
 *
 * @see https://next-auth.js.org/configuration/options
 */
export const authConfig = {
  providers: [
    CredentialsProvider({
      name: "credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null;
        }

        const email = credentials.email as string;
        const password = credentials.password as string;

        const user = await db.user.findUnique({
          where: { email },
        });

        if (!user) {
          return null;
        }

        // OAuth-created accounts have no password — credentials login not allowed
        if (!user.password) return null;

        const passwordMatch = await comparePasswords(password, user.password);
        if (!passwordMatch) return null;

        return user;
      },
    }),
  ],
  session: { strategy: "jwt" },
  adapter: PrismaAdapter(db),
  callbacks: {
    session: async ({ session, token }) => {
      // Guard against stale JWTs pointing to deleted/wiped users.
      // If the user no longer exists in the DB, return an empty session
      // so NextAuth middleware redirects to login instead of 500ing downstream.
      if (token.sub) {
        const exists = await db.user.findUnique({
          where: { id: token.sub },
          select: { id: true, isAdmin: true },
        });
        if (!exists) {
          // Return a session with no user — NextAuth will treat this as unauthenticated
          return { ...session, user: undefined as unknown as typeof session.user };
        }
        // Refresh isAdmin from DB on each session check (handles role changes)
        return {
          ...session,
          user: {
            ...session.user,
            id: token.sub,
            isAdmin: exists.isAdmin ?? false,
          },
        };
      }
      return session;
    },
    jwt: ({ token, user }) => {
      // Persist isAdmin into the JWT so the session callback can read it
      if (user) {
        // user here is the DB record returned by authorize()
        const dbUser = user as { isAdmin?: boolean };
        token.isAdmin = dbUser.isAdmin ?? false;
      }
      return token;
    },
  },
} satisfies NextAuthConfig;
