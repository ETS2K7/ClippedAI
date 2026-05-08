"use client";

import { useState, useEffect } from "react";
import useSWR from "swr";
import { useSession } from "~/lib/auth-client";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Button } from "~/components/ui/button";
import { Skeleton } from "~/components/ui/skeleton";
import { fetcher } from "~/lib/fetcher";
import AppShell from "~/components/app-shell";
import {
  Users,
  Video,
  Clapperboard,
  Shield,
  ShieldAlert,
  ArrowLeft,
  Trash2,
} from "lucide-react";
import Link from "next/link";

interface AdminStats {
  totalUsers: number;
  totalTasks: number;
  totalClips: number;
}

interface AdminUser {
  id: string;
  email: string;
  name: string | null;
  isAdmin: boolean;
  emailVerified: string | null;
  _count: {
    uploadedFiles: number;
    clips: number;
  };
}

export default function AdminPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const isSuperAdmin = session?.user?.email === "ebelthomasseiko@gmail.com";

  const { data: stats, error: statsError } = useSWR<AdminStats>(
    isSuperAdmin ? "/api/admin/stats" : null,
    fetcher,
  );

  const {
    data: users,
    error: usersError,
    mutate: mutateUsers,
  } = useSWR<AdminUser[]>(
    isSuperAdmin ? "/api/admin/users" : null,
    fetcher,
  );

  const [togglingAdminId, setTogglingAdminId] = useState<string | null>(null);
  const [deletingUserId, setDeletingUserId] = useState<string | null>(null);

  // Security guard redirect safely in useEffect to prevent render violation
  useEffect(() => {
    if (!isPending && !isSuperAdmin) {
      if (typeof window !== "undefined") {
        router.push("/dashboard");
      }
    }
  }, [isPending, session, router]);

  if (!isPending && !session?.user?.isAdmin) {
    return (
      <div className="flex h-screen items-center justify-center bg-black">
        <Skeleton className="h-12 w-12 rounded-full bg-white/20" />
      </div>
    );
  }

  const toggleAdminStatus = async (user: AdminUser) => {
    if (user.id === session?.user?.id) {
      alert("You cannot modify your own admin status.");
      return;
    }

    setTogglingAdminId(user.id);
    try {
      const res = await fetch(`/api/admin/users/${user.id}/admin`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_admin: !user.isAdmin }),
      });

      if (!res.ok) {
        const data = (await res.json()) as { error?: string };
        alert(data.error ?? "Failed to toggle admin status.");
      } else {
        await mutateUsers(
          users?.map((u) =>
            u.id === user.id ? { ...u, isAdmin: !user.isAdmin } : u,
          ),
          { revalidate: false },
        );
      }
    } catch (err) {
      console.error(err);
      alert("A network error occurred.");
    } finally {
      setTogglingAdminId(null);
    }
  };

  const deleteUser = async (user: AdminUser) => {
    if (user.id === session?.user?.id) {
      alert("You cannot delete your own account.");
      return;
    }

    if (
      !confirm(
        `Are you absolutely sure you want to permanently delete ${user.email}?\nThis action cannot be undone.`,
      )
    ) {
      return;
    }

    setDeletingUserId(user.id);
    try {
      const res = await fetch(`/api/admin/users/${user.id}`, {
        method: "DELETE",
      });

      if (!res.ok) {
        const data = (await res.json()) as { error?: string };
        alert(data.error ?? "Failed to delete user.");
      } else {
        await mutateUsers(
          users?.filter((u) => u.id !== user.id),
          { revalidate: false },
        );
      }
    } catch (err) {
      console.error(err);
      alert("A network error occurred.");
    } finally {
      setDeletingUserId(null);
    }
  };

  return (
    <AppShell>
      <div className="min-h-screen pb-20">
        <div className="mx-auto max-w-7xl space-y-8 px-4 py-8 sm:px-6 lg:px-8">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-white/10 pb-6">
            <div className="space-y-1">
              <div className="flex items-center gap-4">
                <Link href="/dashboard">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 rounded-md text-white/40 hover:text-white"
                  >
                    <ArrowLeft className="h-5 w-5" />
                  </Button>
                </Link>
                <h1 className="font-syne text-3xl font-black tracking-tighter text-white uppercase">
                  SYSTEM ADMIN
                </h1>
              </div>
              <p className="pl-12 font-mono text-xs tracking-widest text-white/40 uppercase">
                Platform Statistics & User Management
              </p>
            </div>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
            <Card className="brutal-card bg-black">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="font-mono text-xs font-bold tracking-widest text-white/50 uppercase">
                  Total Users
                </CardTitle>
                <Users className="h-4 w-4 text-white/50" />
              </CardHeader>
              <CardContent>
                {!stats && !statsError ? (
                  <Skeleton className="h-10 w-24" />
                ) : (
                  <div className="font-syne text-4xl font-black text-white">
                    {stats?.totalUsers.toLocaleString() ?? "0"}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="brutal-card bg-black">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="font-mono text-xs font-bold tracking-widest text-white/50 uppercase">
                  Raw Videos
                </CardTitle>
                <Video className="h-4 w-4 text-white/50" />
              </CardHeader>
              <CardContent>
                {!stats && !statsError ? (
                  <Skeleton className="h-10 w-24" />
                ) : (
                  <div className="font-syne text-4xl font-black text-white">
                    {stats?.totalTasks.toLocaleString() ?? "0"}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="brutal-card bg-black">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="font-mono text-xs font-bold tracking-widest text-white/50 uppercase">
                  Generations
                </CardTitle>
                <Clapperboard className="h-4 w-4 text-white/50" />
              </CardHeader>
              <CardContent>
                {!stats && !statsError ? (
                  <Skeleton className="h-10 w-24" />
                ) : (
                  <div className="font-syne bg-gradient-to-r from-red-400 to-[#FFE600] bg-clip-text text-4xl font-black text-[url('/textures/noise.png')] text-transparent">
                    {stats?.totalClips.toLocaleString() ?? "0"}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* User Data Table */}
          <Card className="brutal-card overflow-hidden border border-white/10 bg-black">
            <CardHeader className="border-b border-white/5 pb-4">
              <CardTitle className="flex items-center gap-2 font-mono text-sm font-bold tracking-widest text-white uppercase">
                <Shield className="h-4 w-4 text-red-500" />
                Directory Access
              </CardTitle>
            </CardHeader>
            <CardContent className="overflow-x-auto p-0">
              {!users && !usersError ? (
                <div className="space-y-4 p-8">
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-8 w-full" />
                </div>
              ) : usersError ? (
                <div className="p-8 font-mono text-sm text-red-500">
                  Failed to load system directory.
                </div>
              ) : (
                <table className="w-full text-left text-sm text-white/70">
                  <thead className="bg-[#111] font-mono text-[10px] tracking-widest text-white/40 uppercase">
                    <tr>
                      <th className="px-6 py-4 font-normal">Account</th>
                      <th className="px-6 py-4 font-normal">Raw Files</th>
                      <th className="px-6 py-4 font-normal">Clips Generated</th>
                      <th className="px-6 py-4 text-right font-normal">
                        Access Level
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5 font-medium">
                    {users?.map((user) => (
                      <tr
                        key={user.id}
                        className="transition-colors hover:bg-white/[0.02]"
                      >
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex flex-col">
                            <span className="text-sm text-white">
                              {user.email}
                            </span>
                            <span className="mt-1 font-mono text-[10px] text-white/30">
                              ID: {user.id}
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap tabular-nums">
                          {user._count.uploadedFiles}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap tabular-nums">
                          {user._count.clips}
                        </td>
                        <td className="space-x-2 px-6 py-4 text-right whitespace-nowrap">
                          <Button
                            variant="outline"
                            size="sm"
                            disabled={
                              togglingAdminId === user.id ||
                              deletingUserId === user.id ||
                              user.id === session?.user?.id
                            }
                            onClick={() => toggleAdminStatus(user)}
                            className={`rounded-md font-mono text-[10px] tracking-widest uppercase transition-all ${
                              user.isAdmin
                                ? "border-red-500/30 bg-red-500/10 text-red-500 hover:bg-red-500/20 hover:text-red-400"
                                : "border-white/10 bg-transparent text-white/50 hover:bg-white/10 hover:text-white"
                            }`}
                          >
                            {togglingAdminId === user.id ? (
                              "Updating..."
                            ) : user.isAdmin ? (
                              <>
                                <ShieldAlert className="mr-2 h-3 w-3" />
                                Admin
                              </>
                            ) : (
                              "Promote"
                            )}
                          </Button>

                          {user.id !== session?.user?.id && (
                            <Button
                              variant="outline"
                              size="sm"
                              disabled={deletingUserId === user.id}
                              onClick={() => deleteUser(user)}
                              className="rounded-md border-red-500/20 bg-transparent font-mono text-[10px] tracking-widest text-red-500/50 uppercase transition-all hover:border-red-500/30 hover:bg-red-500/10 hover:text-red-500"
                            >
                              {deletingUserId === user.id ? (
                                "..."
                              ) : (
                                <Trash2 className="h-4 w-4" />
                              )}
                            </Button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </AppShell>
  );
}
