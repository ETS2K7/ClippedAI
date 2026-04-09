"use client";

import { useState } from "react";
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
  ArrowLeft
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

  const { data: stats, error: statsError } = useSWR<AdminStats>(
    session?.user?.isAdmin ? "/api/admin/stats" : null,
    fetcher
  );

  
  const { data: users, error: usersError, mutate: mutateUsers } = useSWR<AdminUser[]>(
    session?.user?.isAdmin ? "/api/admin/users" : null,
    fetcher
  );

  const [togglingAdminId, setTogglingAdminId] = useState<string | null>(null);

  // Security guard redirect
  if (!isPending && !session?.user?.isAdmin) {
    if (typeof window !== "undefined") {
      router.push("/dashboard");
    }
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
        const data = await res.json() as { error?: string };
        alert(data.error ?? "Failed to toggle admin status.");
      } else {
        await mutateUsers();
      }
    } catch (err) {
      console.error(err);
      alert("A network error occurred.");
    } finally {
      setTogglingAdminId(null);
    }
  };

  return (
    <AppShell>
      <div className="min-h-screen pb-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
          
          {/* Header */}
          <div className="flex items-center justify-between border-b border-white/10 pb-6">
            <div className="space-y-1">
              <div className="flex items-center gap-4">
                <Link href="/dashboard">
                  <Button variant="ghost" size="icon" className="h-8 w-8 text-white/40 hover:text-white rounded-md">
                    <ArrowLeft className="h-5 w-5" />
                  </Button>
                </Link>
                <h1 className="text-3xl font-black font-syne uppercase tracking-tighter text-white">SYSTEM ADMIN</h1>
              </div>
              <p className="text-white/40 text-xs font-mono uppercase tracking-widest pl-12">
                Platform Statistics & User Management
              </p>
            </div>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="brutal-card bg-black">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-xs font-mono font-bold tracking-widest text-white/50 uppercase">
                  Total Users
                </CardTitle>
                <Users className="w-4 h-4 text-white/50" />
              </CardHeader>
              <CardContent>
                {!stats && !statsError ? (
                  <Skeleton className="h-10 w-24" />
                ) : (
                  <div className="text-4xl font-black font-syne text-white">
                    {stats?.totalUsers.toLocaleString() ?? "0"}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="brutal-card bg-black">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-xs font-mono font-bold tracking-widest text-white/50 uppercase">
                  Raw Videos
                </CardTitle>
                <Video className="w-4 h-4 text-white/50" />
              </CardHeader>
              <CardContent>
                {!stats && !statsError ? (
                  <Skeleton className="h-10 w-24" />
                ) : (
                  <div className="text-4xl font-black font-syne text-white">
                    {stats?.totalTasks.toLocaleString() ?? "0"}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="brutal-card bg-black">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-xs font-mono font-bold tracking-widest text-white/50 uppercase">
                  Generations
                </CardTitle>
                <Clapperboard className="w-4 h-4 text-white/50" />
              </CardHeader>
              <CardContent>
                {!stats && !statsError ? (
                  <Skeleton className="h-10 w-24" />
                ) : (
                  <div className="text-4xl font-black font-syne text-[url('/textures/noise.png')] text-transparent bg-clip-text bg-gradient-to-r from-red-400 to-[#FFE600]">
                    {stats?.totalClips.toLocaleString() ?? "0"}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* User Data Table */}
          <Card className="brutal-card bg-black border border-white/10 overflow-hidden">
            <CardHeader className="border-b border-white/5 pb-4">
              <CardTitle className="text-sm font-bold font-mono tracking-widest text-white uppercase flex items-center gap-2">
                <Shield className="w-4 h-4 text-red-500" />
                Directory Access
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0 overflow-x-auto">
              {!users && !usersError ? (
                <div className="p-8 space-y-4">
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-8 w-full" />
                </div>
              ) : usersError ? (
                <div className="p-8 text-red-500 font-mono text-sm">Failed to load system directory.</div>
              ) : (
                <table className="w-full text-left text-sm text-white/70">
                  <thead className="bg-[#111] font-mono text-[10px] uppercase tracking-widest text-white/40">
                    <tr>
                      <th className="px-6 py-4 font-normal">Account</th>
                      <th className="px-6 py-4 font-normal">Raw Files</th>
                      <th className="px-6 py-4 font-normal">Clips Generated</th>
                      <th className="px-6 py-4 font-normal text-right">Access Level</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5 font-medium">
                    {users?.map((user) => (
                      <tr key={user.id} className="hover:bg-white/[0.02] transition-colors">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex flex-col">
                            <span className="text-white text-sm">{user.email}</span>
                            <span className="text-white/30 text-[10px] font-mono mt-1">ID: {user.id}</span>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap tabular-nums">
                          {user._count.uploadedFiles}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap tabular-nums">
                          {user._count.clips}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-right">
                          <Button
                            variant="outline"
                            size="sm"
                            disabled={togglingAdminId === user.id || user.id === session?.user?.id}
                            onClick={() => toggleAdminStatus(user)}
                            className={`rounded-md font-mono text-[10px] uppercase tracking-widest transition-all ${
                              user.isAdmin 
                                ? 'bg-red-500/10 text-red-500 border-red-500/30 hover:bg-red-500/20 hover:text-red-400' 
                                : 'bg-transparent text-white/50 border-white/10 hover:bg-white/10 hover:text-white'
                            }`}
                          >
                            {togglingAdminId === user.id ? (
                              "Updating..."
                            ) : user.isAdmin ? (
                              <>
                                <ShieldAlert className="w-3 h-3 mr-2" />
                                Admin
                              </>
                            ) : (
                              "Promote"
                            )}
                          </Button>
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
