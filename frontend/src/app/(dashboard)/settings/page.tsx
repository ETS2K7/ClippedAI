"use client";

import { useState, useEffect } from "react";
import { Button } from "~/components/ui/button";
import { Alert, AlertDescription } from "~/components/ui/alert";
import { Skeleton } from "~/components/ui/skeleton";
import { Switch } from "~/components/ui/switch";
import { Label } from "~/components/ui/label";
import { Separator } from "~/components/ui/separator";
import { useSession } from "~/lib/auth-client";
import Link from "next/link";
import { CheckCircle, AlertCircle, ArrowLeft, Mail } from "lucide-react";
import useSWR from "swr";
import { fetcher } from "~/lib/fetcher";

export default function SettingsPage() {
  const [completionEmails, setCompletionEmails] = useState(true);
  const [success, setSuccess] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { data: session, isPending } = useSession();

  const swrOptions = { revalidateOnFocus: false };
  const {
    data: prefsData,
    error: prefsError,
    mutate: mutatePrefs,
  } = useSWR(session?.user ? "/api/preferences" : null, fetcher, swrOptions);

  const isFetching = session?.user && !prefsData && !prefsError;

  useEffect(() => {
    if (prefsData) {
      setCompletionEmails(prefsData.notifyOnCompletion ?? true);
    }
  }, [prefsData]);

  const handleSavePreferences = async () => {
    setIsLoading(true);
    setError(null);
    setSuccess(false);
    try {
      const response = await fetch("/api/preferences", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ notifyOnCompletion: completionEmails }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to save preferences");
      }
      setSuccess(true);
      mutatePrefs();
      setTimeout(() => setSuccess(false), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save preferences");
    } finally {
      setIsLoading(false);
    }
  };

  if (isPending || isFetching) {
    return (
      <div className="flex min-h-screen items-center justify-center p-4">
        <div className="space-y-4">
          <Skeleton className="mx-auto h-4 w-32 rounded-md bg-white/[0.1]" />
          <Skeleton className="mx-auto h-4 w-48 rounded-md bg-white/[0.1]" />
        </div>
      </div>
    );
  }

  if (!session?.user) {
    return (
      <div className="min-h-screen">
        <div className="mx-auto max-w-4xl px-4 py-24 text-center">
          <h1 className="font-syne mb-4 text-4xl font-black text-white uppercase md:text-5xl">
            SIGN IN REQUIRED.
          </h1>
          <p className="mb-8 font-mono text-xs tracking-widest text-white/40 uppercase">
            You need to sign in to access your settings.
          </p>
          <Link href="/auth/oauth/login">
            <Button
              size="lg"
              className="font-syne rounded-xl bg-white font-black tracking-widest text-black uppercase hover:bg-white/90"
            >
              Sign In
            </Button>
          </Link>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="min-h-screen">
        {/* ── Page header ── */}
        <div className="border-b border-white/[0.1]">
          <div className="w-full pb-8">
            <div className="mb-2 flex items-center gap-3">
              <h1 className="font-syne text-3xl leading-none font-black tracking-tighter text-transparent bg-clip-text bg-gradient-to-b from-white to-white/60 uppercase sm:text-4xl md:text-5xl">
                Settings.
              </h1>
            </div>
            <p className="mt-3 font-mono text-[10px] tracking-widest text-white/40 uppercase sm:mt-4 sm:text-xs">
              Manage your account preferences.
            </p>
          </div>
        </div>

        {/* ── Main content ── */}
        <div className="relative mx-auto max-w-3xl py-8">
          <div className="mx-auto max-w-xl space-y-6 sm:space-y-8">

            {/* ── Notifications ── */}
            <div className="brutal-card space-y-5 p-4 sm:p-6">
              <div>
                <h3 className="mb-2 font-mono text-xs font-bold tracking-widest text-white uppercase sm:text-[14px]">
                  NOTIFICATIONS
                </h3>
                <p className="font-mono text-[10px] tracking-wider text-white/35 uppercase sm:text-xs">
                  Manage how you receive updates about your clips.
                </p>
              </div>

              <div className="flex flex-col justify-between gap-4 rounded-xl border border-white/10 bg-transparent p-4 sm:flex-row sm:items-center">
                <Label
                  htmlFor="completion-emails"
                  className="flex cursor-pointer items-start gap-3 text-sm font-medium text-white/80 sm:items-center"
                >
                  <Mail className="mt-1 h-5 w-5 text-white opacity-80 sm:mt-0" />
                  <div>
                    <span className="font-mono text-xs font-bold tracking-widest uppercase">
                      COMPLETION EMAILS
                    </span>
                    <p className="mt-1 font-mono text-[10px] tracking-wider text-white/40 uppercase">
                      Get notified when clips are ready
                    </p>
                  </div>
                </Label>
                <Switch
                  id="completion-emails"
                  checked={completionEmails}
                  onCheckedChange={setCompletionEmails}
                  disabled={isLoading}
                />
              </div>
            </div>

            <Separator className="bg-white/[0.06]" />

            {success && (
              <Alert className="border-emerald-500/20 bg-emerald-500/5">
                <CheckCircle className="h-4 w-4 text-emerald-400" />
                <AlertDescription className="text-sm text-emerald-400">
                  Preferences saved successfully!
                </AlertDescription>
              </Alert>
            )}

            {error && (
              <Alert className="border-red-500/20 bg-red-500/5">
                <AlertCircle className="h-4 w-4 text-red-400" />
                <AlertDescription className="text-sm text-red-400">
                  {error}
                </AlertDescription>
              </Alert>
            )}

            <Button
              onClick={handleSavePreferences}
              disabled={isLoading}
              className="font-syne h-12 w-full rounded-xl bg-white text-sm font-black tracking-wider text-black uppercase transition-all hover:bg-white/90 disabled:opacity-50 sm:h-14 sm:text-base sm:tracking-widest"
            >
              {isLoading ? "SAVING..." : "SAVE PREFERENCES."}
            </Button>
          </div>
        </div>
      </div>
    </>
  );
}
