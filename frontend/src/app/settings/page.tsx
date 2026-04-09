"use client";

import { useState, useEffect } from "react";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";
import { Separator } from "~/components/ui/separator";
import { Alert, AlertDescription } from "~/components/ui/alert";
import { Skeleton } from "~/components/ui/skeleton";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "~/components/ui/select";
import { Slider } from "~/components/ui/slider";
import { Switch } from "~/components/ui/switch";
import { useSession } from "~/lib/auth-client";
import { track } from "~/lib/datafast";
import Link from "next/link";
import { Type, Palette, CheckCircle, AlertCircle, Settings, ArrowLeft, Mail } from "lucide-react";
import AppShell from "~/components/app-shell";

interface UserPreferences {
  fontFamily: string;
  fontSize: number;
  fontColor: string;
  notifyOnCompletion: boolean;
}


export default function SettingsPage() {
  const [fontFamily, setFontFamily] = useState("TikTokSans-Regular");
  const [fontSize, setFontSize] = useState(24);
  const [fontColor, setFontColor] = useState("#FFFFFF");
  const [completionEmails, setCompletionEmails] = useState(true);
  const [availableFonts, setAvailableFonts] = useState<Array<{ name: string, display_name: string }>>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isFetching, setIsFetching] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const { data: session, isPending } = useSession();

  // Load available fonts from backend and inject them into the page
  useEffect(() => {
    const loadFonts = async () => {
      try {
        const response = await fetch('/api/fonts', { cache: 'no-store' });
        if (response.ok) {
          const data = await response.json();
          setAvailableFonts(data.fonts || []);

          // Dynamically load fonts using @font-face
          const fontFaceStyles = data.fonts.map((font: { name: string }) => {
            return `
              @font-face {
                font-family: '${font.name}';
                src: url('/api/fonts/${font.name}') format('truetype');
                font-weight: normal;
                font-style: normal;
              }
            `;
          }).join('\n');

          // Inject font styles into the page
          const styleElement = document.createElement('style');
          styleElement.id = 'custom-fonts';
          styleElement.innerHTML = fontFaceStyles;

          // Remove existing custom fonts style if present
          const existingStyle = document.getElementById('custom-fonts');
          if (existingStyle) {
            existingStyle.remove();
          }

          document.head.appendChild(styleElement);
        }
      } catch (error) {
        console.error('Failed to load fonts:', error);
      }
    };

    loadFonts();
  }, []);

  // Load user preferences
  useEffect(() => {
    const loadPreferences = async () => {
      if (!session?.user?.id) return;

      setIsFetching(true);
      try {
        const response = await fetch('/api/preferences');
        if (response.ok) {
          const data: UserPreferences = await response.json();
          setFontFamily(data.fontFamily);
          setFontSize(data.fontSize);
          setFontColor(data.fontColor);
          setCompletionEmails(data.notifyOnCompletion ?? true);
        }
      } catch (error) {
        console.error('Failed to load preferences:', error);
      } finally {
        setIsFetching(false);
      }
    };

    loadPreferences();
  }, [session?.user?.id]);

  const handleSavePreferences = async () => {
    setIsLoading(true);
    setError(null);
    setSuccess(false);

    try {
      const response = await fetch('/api/preferences', {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
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
        throw new Error(errorData.error || 'Failed to save preferences');
      }

      track("preferences_saved");
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (error) {
      console.error('Error saving preferences:', error);
      setError(error instanceof Error ? error.message : 'Failed to save preferences');
    } finally {
      setIsLoading(false);
    }
  };

  if (isPending || isFetching) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <div className="space-y-4">
          <Skeleton className="h-4 w-32 mx-auto rounded-md bg-white/[0.1]" />
          <Skeleton className="h-4 w-48 mx-auto rounded-md bg-white/[0.1]" />
          <Skeleton className="h-4 w-24 mx-auto rounded-md bg-white/[0.1]" />
        </div>
      </div>
    );
  }

  if (!session?.user) {
    return (
      <div className="min-h-screen">
        <div className="max-w-4xl mx-auto px-4 py-24 text-center">
          <h1 className="text-4xl md:text-5xl font-black font-syne uppercase text-white mb-4">SIGN IN REQUIRED.</h1>
          <p className="text-white/40 mb-8 font-mono tracking-widest text-xs uppercase">You need to sign in to access your settings.</p>
          <Link href="/login">
            <Button size="lg" className="bg-white hover:bg-white/90 text-black font-black uppercase font-syne tracking-widest rounded-xl">Sign In</Button>
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
          <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8">
            <div className="flex items-center gap-3 mb-6">
              <Link href="/dashboard">
                <Button variant="ghost" size="sm" className="text-white/40 hover:text-white hover:bg-white/[0.06] rounded-full font-mono tracking-widest uppercase text-[10px]">
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  BACK
                </Button>
              </Link>
            </div>
            <div className="flex items-center gap-3 mb-2">
              <h1 className="text-3xl sm:text-4xl md:text-5xl font-black font-syne uppercase tracking-tighter text-white leading-none">SETTINGS.</h1>
            </div>
            <p className="text-[10px] sm:text-xs text-white/40 font-mono tracking-widest uppercase mt-3 sm:mt-4">
              Configure your default preferences for video clip generation.
            </p>
          </div>
        </div>

        {/* ── Main content ── */}
        <div className="relative max-w-3xl mx-auto px-4 sm:px-6 py-8">
          <div className="max-w-xl mx-auto space-y-6 sm:space-y-8">

            {/* ── Font Preferences ── */}
            <div className="brutal-card p-4 sm:p-6 space-y-4 sm:space-y-6">
              <div>
                <h3 className="text-xs sm:text-[14px] font-bold font-mono tracking-widest uppercase text-white mb-2">
                  DEFAULT FONT SETTINGS
                </h3>
                <p className="text-[10px] sm:text-xs text-white/40 font-mono uppercase tracking-wider">
                  These settings will be applied to all new video processing tasks.
                </p>
              </div>

              {/* Font Family Selector */}
              <div className="space-y-2">
                <Label className="text-[10px] font-bold font-mono tracking-widest uppercase text-white/50">
                  Font Family
                </Label>
                <Select value={fontFamily} onValueChange={setFontFamily} disabled={isLoading}>
                  <SelectTrigger className="w-full brutal-input">
                    <SelectValue placeholder="Select font" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableFonts.map((font) => (
                      <SelectItem key={font.name} value={font.name}>
                        <span style={{ fontFamily: `'${font.name}', system-ui, sans-serif` }}>
                          {font.display_name}
                        </span>
                      </SelectItem>
                    ))}
                    {availableFonts.length === 0 && (
                      <SelectItem value="TikTokSans-Regular">TikTok Sans Regular</SelectItem>
                    )}
                  </SelectContent>
                </Select>
              </div>

              {/* Font Size Slider */}
              <div className="space-y-2">
                <Label className="text-[10px] font-bold font-mono tracking-widest uppercase text-white/50">
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
                <Label className="text-[10px] font-bold font-mono tracking-widest uppercase text-white/50">
                  Font Color
                </Label>
                <div className="flex items-center gap-2">
                  <input
                    type="color"
                    value={fontColor}
                    onChange={(e) => setFontColor(e.target.value)}
                    disabled={isLoading}
                    className="w-10 h-8 rounded-md border border-white/10 cursor-pointer disabled:cursor-not-allowed bg-transparent"
                  />
                  <Input
                    type="text"
                    value={fontColor}
                    onChange={(e) => setFontColor(e.target.value)}
                    disabled={isLoading}
                    placeholder="#FFFFFF"
                    className="flex-1 h-9 brutal-input font-mono uppercase"
                    pattern="^#[0-9A-Fa-f]{6}$"
                  />
                </div>
                <div className="flex gap-1.5 mt-1">
                  {["#FFFFFF", "#000000", "#FFD700", "#FF6B6B", "#4ECDC4", "#45B7D1"].map((color) => (
                    <button
                      key={color}
                      type="button"
                      onClick={() => setFontColor(color)}
                      disabled={isLoading}
                      className="w-6 h-6 rounded-full border-2 border-white/10 cursor-pointer hover:scale-125 hover:border-white/30 transition-all disabled:cursor-not-allowed"
                      style={{ backgroundColor: color }}
                      title={color}
                    />
                  ))}
                </div>
              </div>

              {/* Preview */}
              <div className="space-y-2">
                <Label className="text-[10px] font-bold font-mono tracking-widest uppercase text-white/50">PREVIEW</Label>
                <div className="p-6 bg-black border border-white/[0.1] flex items-center justify-center min-h-[100px]">
                  <p
                    style={{
                      color: fontColor,
                      fontSize: `${Math.min(fontSize, 32)}px`,
                      fontFamily: `'${fontFamily}', system-ui, -apple-system, sans-serif`,
                      textAlign: 'center',
                      lineHeight: '1.4'
                    }}
                    className="font-medium"
                  >
                    Your subtitle will look like this
                  </p>
                </div>
              </div>
            </div>

            {/* ── Notifications ── */}
            <div className="brutal-card p-4 sm:p-6 space-y-5">
              <div>
                <h3 className="text-xs sm:text-[14px] font-bold font-mono tracking-widest uppercase text-white mb-2">
                  NOTIFICATIONS
                </h3>
                <p className="text-[10px] sm:text-xs text-white/35 font-mono uppercase tracking-wider">
                  Manage how you receive updates about your clips.
                </p>
              </div>

              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 border border-white/10 bg-transparent rounded-xl">
                <Label htmlFor="completion-emails" className="flex items-start gap-3 sm:items-center text-sm font-medium text-white/80 cursor-pointer">
                  <Mail className="w-5 h-5 text-white mt-1 sm:mt-0 opacity-80" />
                  <div>
                    <span className="font-mono tracking-widest uppercase font-bold text-xs">COMPLETION EMAILS</span>
                    <p className="text-[10px] text-white/40 font-mono tracking-wider uppercase mt-1">Get notified when clips are ready</p>
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
              className="w-full h-12 sm:h-14 text-sm sm:text-base rounded-xl bg-white hover:bg-white/90 text-black font-black uppercase font-syne tracking-wider sm:tracking-widest transition-all disabled:opacity-50"
            >
              {isLoading ? "SAVING..." : "SAVE PREFERENCES."}
            </Button>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
