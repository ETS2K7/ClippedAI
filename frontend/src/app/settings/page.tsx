"use client";

import { useState, useEffect } from "react";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";
import { Separator } from "~/components/ui/separator";
import { Alert, AlertDescription } from "~/components/ui/alert";
import { Skeleton } from "~/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { Slider } from "~/components/ui/slider";
import { Switch } from "~/components/ui/switch";
import { useSession } from "~/lib/auth-client";
import { track } from "~/lib/datafast";
import Link from "next/link";
import { CheckCircle, AlertCircle, ArrowLeft, Mail } from "lucide-react";
import AppShell from "~/components/app-shell";
import useSWR from "swr";
import { fetcher } from "~/lib/fetcher";

export default function SettingsPage() {
  const [fontFamily, setFontFamily] = useState("TikTokSans-Regular");
  const [fontSize, setFontSize] = useState(24);
  const [fontColor, setFontColor] = useState("#FFFFFF");
  const [completionEmails, setCompletionEmails] = useState(true);
  const [success, setSuccess] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { data: session, isPending } = useSession();

  // SWR: Global Data Fetching
  const swrOptions = { revalidateOnFocus: false };
  const { data: fontsData } = useSWR(
    session?.user ? "/api/fonts" : null,
    fetcher,
    swrOptions,
  );
  const {
    data: prefsData,
    error: prefsError,
    mutate: mutatePrefs,
  } = useSWR(session?.user ? "/api/preferences" : null, fetcher, swrOptions);

  const availableFonts: Array<{ name: string; display_name: string }> =
    fontsData?.fonts || [];
  const isFetching = session?.user && !prefsData && !prefsError;

  // On preferences loaded, set initial local values
  useEffect(() => {
    if (prefsData) {
      setFontFamily(prefsData.fontFamily || "TikTokSans-Regular");
      setFontSize(prefsData.fontSize || 24);
      setFontColor(prefsData.fontColor || "#FFFFFF");
      setCompletionEmails(prefsData.notifyOnCompletion ?? true);
    }
  }, [prefsData]);

  // Inject font-faces globally based on SWR fonts
  useEffect(() => {
    if (availableFonts.length > 0) {
      const fontFaceStyles = availableFonts
        .map((font) => {
          return `
          @font-face {
            font-family: '${font.name}';
            src: url('/api/fonts/${font.name}') format('truetype');
            font-weight: normal;
            font-style: normal;
          }
        `;
        })
        .join("\n");

      let styleElement = document.getElementById("custom-fonts");
      if (!styleElement) {
        styleElement = document.createElement("style");
        styleElement.id = "custom-fonts";
        document.head.appendChild(styleElement);
      }
      styleElement.innerHTML = fontFaceStyles;
    }
  }, [availableFonts]);

  const handleSavePreferences = async () => {
    setIsLoading(true);
    setError(null);
    setSuccess(false);

    try {
      const response = await fetch("/api/preferences", {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          fontFamily,
          fontSize,
          fontColor,
          notifyOnCompletion: completionEmails,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to save preferences");
      }

      track("preferences_saved");
      setSuccess(true);
      mutatePrefs();
      setTimeout(() => setSuccess(false), 3000);
    } catch (error) {
      console.error("Error saving preferences:", error);
      setError(
        error instanceof Error ? error.message : "Failed to save preferences",
      );
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
          <Skeleton className="mx-auto h-4 w-24 rounded-md bg-white/[0.1]" />
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
          <Link href="/login">
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
    <AppShell>
      <div className="min-h-screen">
        {/* ── Page header ── */}
        <div className="border-b border-white/[0.1]">
          <div className="mx-auto max-w-3xl px-4 py-8 sm:px-6">
            <div className="mb-6 flex items-center gap-3">
              <Link href="/dashboard">
                <Button
                  variant="ghost"
                  size="sm"
                  className="rounded-full font-mono text-[10px] tracking-widest text-white/40 uppercase hover:bg-white/[0.06] hover:text-white"
                >
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  BACK
                </Button>
              </Link>
            </div>
            <div className="mb-2 flex items-center gap-3">
              <h1 className="font-syne text-3xl leading-none font-black tracking-tighter text-white uppercase sm:text-4xl md:text-5xl">
                SETTINGS.
              </h1>
            </div>
            <p className="mt-3 font-mono text-[10px] tracking-widest text-white/40 uppercase sm:mt-4 sm:text-xs">
              Configure your default preferences for video clip generation.
            </p>
          </div>
        </div>

        {/* ── Main content ── */}
        <div className="relative mx-auto max-w-3xl px-4 py-8 sm:px-6">
          <div className="mx-auto max-w-xl space-y-6 sm:space-y-8">
            {/* ── Font Preferences ── */}
            <div className="brutal-card space-y-4 p-4 sm:space-y-6 sm:p-6">
              <div>
                <h3 className="mb-2 font-mono text-xs font-bold tracking-widest text-white uppercase sm:text-[14px]">
                  DEFAULT FONT SETTINGS
                </h3>
                <p className="font-mono text-[10px] tracking-wider text-white/40 uppercase sm:text-xs">
                  These settings will be applied to all new video processing
                  tasks.
                </p>
              </div>

              {/* Font Family Selector */}
              <div className="space-y-2">
                <Label className="font-mono text-[10px] font-bold tracking-widest text-white/50 uppercase">
                  Font Family
                </Label>
                <Select
                  value={fontFamily}
                  onValueChange={setFontFamily}
                  disabled={isLoading}
                >
                  <SelectTrigger className="brutal-input w-full">
                    <SelectValue placeholder="Select font" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableFonts.map((font) => (
                      <SelectItem key={font.name} value={font.name}>
                        <span
                          style={{
                            fontFamily: `'${font.name}', system-ui, sans-serif`,
                          }}
                        >
                          {font.display_name}
                        </span>
                      </SelectItem>
                    ))}
                    {availableFonts.length === 0 && (
                      <SelectItem value="TikTokSans-Regular">
                        TikTok Sans Regular
                      </SelectItem>
                    )}
                  </SelectContent>
                </Select>
              </div>

              {/* Font Size Slider */}
              <div className="space-y-2">
                <Label className="font-mono text-[10px] font-bold tracking-widest text-white/50 uppercase">
                  Font Size: {fontSize}px
                </Label>
                <div className="px-1">
                  <Slider
                    value={[fontSize]}
                    onValueChange={(value) => setFontSize(value[0])}
                    max={48}
                    min={12}
                    step={2}
                    disabled={isLoading}
                    className="w-full"
                  />
                </div>
                <div className="flex justify-between text-xs text-white/20">
                  <span>12px</span>
                  <span>48px</span>
                </div>
              </div>

              {/* Font Color Picker */}
              <div className="space-y-2">
                <Label className="font-mono text-[10px] font-bold tracking-widest text-white/50 uppercase">
                  Font Color
                </Label>
                <div className="flex items-center gap-2">
                  <input
                    type="color"
                    value={fontColor}
                    onChange={(e) => setFontColor(e.target.value)}
                    disabled={isLoading}
                    className="h-8 w-10 cursor-pointer rounded-md border border-white/10 bg-transparent disabled:cursor-not-allowed"
                  />
                  <Input
                    type="text"
                    value={fontColor}
                    onChange={(e) => setFontColor(e.target.value)}
                    disabled={isLoading}
                    placeholder="#FFFFFF"
                    className="brutal-input h-9 flex-1 font-mono uppercase"
                    pattern="^#[0-9A-Fa-f]{6}$"
                  />
                </div>
                <div className="mt-1 flex gap-1.5">
                  {[
                    "#FFFFFF",
                    "#000000",
                    "#FFD700",
                    "#FF6B6B",
                    "#4ECDC4",
                    "#45B7D1",
                  ].map((color) => (
                    <button
                      key={color}
                      type="button"
                      onClick={() => setFontColor(color)}
                      disabled={isLoading}
                      className="h-6 w-6 cursor-pointer rounded-full border-2 border-white/10 transition-all hover:scale-125 hover:border-white/30 disabled:cursor-not-allowed"
                      style={{ backgroundColor: color }}
                      title={color}
                    />
                  ))}
                </div>
              </div>

              {/* Preview */}
              <div className="space-y-2">
                <Label className="font-mono text-[10px] font-bold tracking-widest text-white/50 uppercase">
                  PREVIEW
                </Label>
                <div className="flex min-h-[100px] items-center justify-center border border-white/[0.1] bg-black p-6">
                  <p
                    style={{
                      color: fontColor,
                      fontSize: `${Math.min(fontSize, 32)}px`,
                      fontFamily: `'${fontFamily}', system-ui, -apple-system, sans-serif`,
                      textAlign: "center",
                      lineHeight: "1.4",
                    }}
                    className="font-medium"
                  >
                    Your subtitle will look like this
                  </p>
                </div>
              </div>
            </div>

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

            {/* Success/Error Messages */}
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

            {/* Save Button */}
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
    </AppShell>
  );
}
