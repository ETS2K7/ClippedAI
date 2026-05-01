import { PrismaAdapter } from "@auth/prisma-adapter";
import { type DefaultSession, type NextAuthConfig } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import GoogleProvider from "next-auth/providers/google";
import { comparePasswords } from "~/lib/auth";

import { db } from "~/server/db";
import { env } from "~/env";

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
    GoogleProvider({
      clientId: env.GOOGLE_CLIENT_ID,
      clientSecret: env.GOOGLE_CLIENT_SECRET,
    }),
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

        // Only return fields needed by NextAuth — never expose the password hash
        return {
          id: user.id,
          email: user.email,
          name: user.name,
          image: user.image,
          isAdmin: user.isAdmin,
        };
      },
    }),
  ],
  session: { strategy: "jwt" },
  adapter: PrismaAdapter(db),
  callbacks: {
    session: async ({ session, token }) => {
      if (token.sub) {
        // isAdmin is baked into the JWT at sign-in (see jwt callback below).
        // Only re-query the DB if the flag is missing from the token entirely,
        // which avoids a DB hit on every single session check (was causing 429s).
        if (typeof token.isAdmin !== "undefined") {
          return {
            ...session,
            user: {
              ...session.user,
              id: token.sub,
              isAdmin: token.isAdmin as boolean,
            },
          };
        }

        // Fallback: token is missing isAdmin (e.g. old sessions pre-migration).
        // Query the DB once, and the next JWT refresh will populate the flag.
        const exists = await db.user.findUnique({
          where: { id: token.sub },
          select: { id: true, isAdmin: true },
        });
        if (!exists) {
          // User was deleted — invalidate session by clearing required fields.
          // This forces NextAuth to treat the session as unauthenticated.
          return {
            ...session,
            user: { ...session.user, id: "", isAdmin: false },
            expires: new Date(0).toISOString(),
          };
        }
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
