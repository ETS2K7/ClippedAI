"use client";

import { useState } from "react";
import Dropzone, { type DropzoneState } from "shadcn-dropzone";
import type { Clip } from "@prisma/client";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { generateUploadUrl } from "~/actions/s3";
import { processVideo } from "~/actions/generation";
import { signOut } from "next-auth/react";

import { Button } from "./ui/button";
import { Avatar, AvatarFallback } from "./ui/avatar";
import { Separator } from "./ui/separator";
import { Alert, AlertDescription } from "./ui/alert";
import { Badge } from "./ui/badge";
import { Progress } from "./ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./ui/table";
import { ClipDisplay } from "./clip-display";

import {
  Youtube,
  CheckCircle,
  AlertCircle,
  Loader2,
  Upload,
  Monitor,
  Menu,
  X,
  LogOut,
  List,
  Film,
  UploadCloud
} from "lucide-react";

export function DashboardClient({
  uploadedFiles,
  clips,
  email,
}: {
  uploadedFiles: {
    id: string;
    s3Key: string;
    filename: string;
    status: string;
    clipsCount: number;
    createdAt: Date;
  }[];
  clips: Clip[];
  email: string;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  
  const [sourceType, setSourceType] = useState<"youtube" | "upload">("upload");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  
  const router = useRouter();
  
  const handleRefresh = async () => {
    setRefreshing(true);
    router.refresh();
    setTimeout(() => setRefreshing(false), 600);
  };

  const handleDrop = (acceptedFiles: File[]) => {
    setFiles(acceptedFiles);
    setError(null);
  };

  const handleSignOut = async () => {
    await signOut({ callbackUrl: "/login" });
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (sourceType === "youtube") {
      setError("YouTube imports are not yet supported in this version. Please upload a file.");
      return;
    }
    
    if (files.length === 0) {
      setError("Please select a file to upload.");
      return;
    }

    const file = files[0]!;
    setUploading(true);
    setError(null);
    setProgress(10);

    try {
      const { success, signedUrl, uploadedFileId } = await generateUploadUrl({
        filename: file.name,
        contentType: file.type,
      });

      if (!success) throw new Error("Failed to get upload URL");
      
      setProgress(30);

      const uploadResponse = await fetch(signedUrl, {
        method: "PUT",
        body: file,
        headers: { "Content-Type": file.type },
      });

      if (!uploadResponse.ok)
        throw new Error(`Upload failed with status: ${uploadResponse.status}`);

      setProgress(80);
      await processVideo(uploadedFileId);
      setProgress(100);
      setFiles([]);

      toast.success("Video uploaded successfully", {
        description: "Your video has been scheduled for processing. Check the queue status below.",
        duration: 5000,
      });
      
      void handleRefresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      toast.error("Upload failed", {
        description: "There was a problem uploading your video. Please try again.",
      });
    } finally {
      setUploading(false);
      setProgress(0);
    }
  };

  const latestFile = uploadedFiles.length > 0 ? uploadedFiles[0] : null;
  const userName = email.split("@")[0] ?? "User";

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <div className="border-b bg-white relative">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-stone-900 flex items-center justify-center">
                 <Film className="w-5 h-5 text-white" />
              </div>
              <h1 className="text-xl font-bold text-black">ClippedAI</h1>
            </div>

            {/* Desktop nav */}
            <div className="hidden md:flex items-center gap-2">
              <Link href="/dashboard">
                <Button variant="outline" size="sm">Dashboard</Button>
              </Link>
              <Button variant="ghost" size="sm" onClick={handleSignOut} className="text-stone-500 hover:text-red-600">
                Sign Out
              </Button>
              <div className="flex items-center gap-3 hover:bg-stone-50 rounded-lg px-2 py-1.5 transition-colors">
                <Avatar className="w-8 h-8 border border-stone-200">
                  <AvatarFallback className="bg-stone-100 text-stone-700 text-sm font-medium">
                    {email.charAt(0).toUpperCase()}
                  </AvatarFallback>
                </Avatar>
                <div className="hidden sm:block text-left">
                  <p className="text-sm font-medium text-stone-900">{userName}</p>
                  <p className="text-xs text-stone-500">{email}</p>
                </div>
              </div>
            </div>

            {/* Mobile hamburger */}
            <div className="flex items-center gap-2 md:hidden">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="p-2"
              >
                {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </Button>
            </div>
          </div>
        </div>

        {/* Mobile menu */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t bg-white absolute left-0 right-0 z-50 shadow-lg">
            <div className="px-4 py-3 space-y-1">
              <div className="flex items-center gap-3 rounded-lg px-3 py-2.5 bg-stone-50">
                <Avatar className="w-8 h-8">
                  <AvatarFallback className="bg-stone-200 text-stone-700 text-sm font-medium">
                    {email.charAt(0).toUpperCase()}
                  </AvatarFallback>
                </Avatar>
                <div className="min-w-0">
                  <p className="text-sm font-medium text-stone-900 truncate">{userName}</p>
                  <p className="text-xs text-stone-500 truncate">{email}</p>
                </div>
              </div>
              <Separator className="my-2" />
              <button
                onClick={() => { setMobileMenuOpen(false); void handleSignOut(); }}
                className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-red-600 hover:bg-red-50 transition-colors w-full text-left font-medium"
              >
                <LogOut className="w-4 h-4" />
                Sign Out
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-6 py-10">
        
        {/* Latest Generation Banner */}
        {latestFile && (
          <div className="block mb-8">
            <div className="flex items-center justify-between p-4 rounded-xl border border-stone-200 bg-stone-50/50 hover:bg-stone-50 transition-colors group">
              <div className="flex items-center gap-4 min-w-0">
                <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-stone-900 flex items-center justify-center">
                  <Film className="w-5 h-5 text-white" />
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-medium text-stone-900 truncate">{latestFile.filename}</p>
                  <div className="flex items-center gap-2 text-xs text-stone-500 mt-0.5">
                    <span>{new Date(latestFile.createdAt).toLocaleDateString()}</span>
                    <span>&middot;</span>
                    <span>{latestFile.clipsCount} {latestFile.clipsCount === 1 ? "clip" : "clips"}</span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-3 flex-shrink-0">
                {latestFile.status === "processed" ? (
                  <Badge className="bg-green-100 text-green-800 text-xs border-green-200">
                    <CheckCircle className="w-3 h-3 mr-1" />Completed
                  </Badge>
                ) : latestFile.status === "processing" || latestFile.status === "queued" ? (
                  <Badge className="bg-blue-100 text-blue-800 text-xs border-blue-200">
                    <Loader2 className="w-3 h-3 animate-spin mr-1" />
                    {latestFile.status === "queued" ? "Queued" : "Processing"}
                  </Badge>
                ) : (
                  <Badge variant="destructive" className="text-xs">{latestFile.status}</Badge>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Two Column Layout */}
        <div className="flex flex-col lg:flex-row gap-10 items-start mb-16">
          <div className="flex-1 min-w-0 w-full">
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-stone-900 mb-2">Create New Clip</h2>
              <p className="text-stone-500">
                Upload a video file — AI will find the best viral moments and extract them automatically.
              </p>
            </div>

            <form onSubmit={handleUpload} className="space-y-6">
              <div className="space-y-3">
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => { setSourceType("upload"); setError(null); }}
                    disabled={uploading}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      sourceType === "upload"
                        ? "bg-stone-900 text-white shadow-sm"
                        : "bg-stone-100 text-stone-600 hover:bg-stone-200"
                    }`}
                  >
                    <Upload className="w-4 h-4" />
                    Upload Video
                  </button>
                  <button
                    type="button"
                    onClick={() => setSourceType("youtube")}
                    disabled={uploading}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all cursor-not-allowed opacity-50 ${
                      sourceType === "youtube"
                        ? "bg-stone-900 text-white shadow-sm"
                        : "bg-stone-100 text-stone-600 hover:bg-stone-200"
                    }`}
                    title="Coming soon"
                  >
                    <Youtube className="w-4 h-4" />
                    YouTube URL (Soon)
                  </button>
                </div>

                {sourceType === "youtube" ? (
                   <div className="relative p-8 border-2 border-dashed border-stone-200 bg-stone-50 rounded-xl text-center">
                     <p className="text-stone-500 text-sm">YouTube importing is coming soon.</p>
                   </div>
                ) : (
                  <Dropzone
                    onDrop={handleDrop}
                    accept={{ "video/mp4": [".mp4"], "video/quicktime": [".mov"], "video/x-msvideo": [".avi"] }}
                    maxSize={500 * 1024 * 1024}
                    disabled={uploading}
                    maxFiles={1}
                  >
                    {(_dropzone: DropzoneState) => (
                      <div className="relative border-2 border-dashed border-stone-300 rounded-xl p-10 text-center hover:border-stone-400 transition-colors cursor-pointer bg-stone-50/50">
                        <UploadCloud className="w-10 h-10 text-stone-400 mx-auto mb-4" />
                        {files.length > 0 ? (
                          <div className="space-y-1">
                            <p className="text-sm font-medium text-stone-900">{files[0]?.name}</p>
                            <p className="text-xs text-stone-500">{(files[0]!.size / (1024*1024)).toFixed(2)} MB</p>
                          </div>
                        ) : (
                          <>
                            <p className="text-sm font-medium text-stone-700 mb-1">Drop a video file here or click to browse</p>
                            <p className="text-xs text-stone-400">MP4, MOV, AVI up to 500MB</p>
                          </>
                        )}
                      </div>
                    )}
                  </Dropzone>
                )}
              </div>

              {uploading && (
                <div className="space-y-4 pt-2">
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-stone-600">Uploading to S3...</span>
                      <span className="text-stone-900 font-medium">{progress}%</span>
                    </div>
                    <Progress value={progress} className="h-2 bg-stone-100" />
                  </div>
                </div>
              )}

              {error && (
                <Alert className="border-red-200 bg-red-50 py-3">
                  <AlertCircle className="h-4 w-4 text-red-500" />
                  <AlertDescription className="text-sm text-red-700 ml-2">{error}</AlertDescription>
                </Alert>
              )}

              <Button
                type="submit"
                className="w-full h-12 text-base rounded-xl bg-stone-900 hover:bg-stone-800"
                disabled={sourceType === "youtube" || files.length === 0 || uploading}
              >
                {uploading ? (
                  <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Processing...</>
                ) : (
                  "Upload & Process Video"
                )}
              </Button>
            </form>
          </div>

          {/* Right Column — Phone Preview */}
          <div className="hidden lg:block flex-shrink-0 overflow-hidden w-[340px]">
            <div className="lg:sticky lg:top-8 w-[340px]">
              <div className="flex items-center justify-center gap-2 mb-5 text-sm text-stone-400 font-medium">
                <Monitor className="w-4 h-4" />
                <span>Responsive Framing</span>
              </div>
              <div className="mx-auto block" style={{ maxWidth: "300px" }}>
                <div className="relative bg-stone-950 p-3 rounded-[3rem] shadow-xl border border-stone-200">
                  <div className="relative overflow-hidden bg-stone-900 rounded-[2.25rem] h-[580px] flex items-center justify-center group border border-stone-800">
                    <div className="absolute inset-0 bg-gradient-to-b from-stone-800 to-stone-900" />
                    <div className="absolute inset-x-8 top-1/4 bottom-1/4 border-2 border-white/20 rounded-xl rounded-bl-3xl border-dashed opacity-50" />
                    <div className="z-10 text-center px-6">
                      <div className="w-16 h-16 mx-auto bg-white/10 rounded-full flex items-center justify-center mb-4">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="opacity-80">
                          <path d="M3 7V5a2 2 0 0 1 2-2h2"></path><path d="M17 3h2a2 2 0 0 1 2 2v2"></path><path d="M21 17v2a2 2 0 0 1-2 2h-2"></path><path d="M7 21H5a2 2 0 0 1-2-2v-2"></path><rect x="7" y="7" width="10" height="10" rx="2"></rect>
                        </svg>
                      </div>
                      <h3 className="text-white font-medium text-lg mb-2 shadow-sm">AI Face Tracking</h3>
                      <p className="text-stone-400 text-xs">ClippedAI automatically centres active speakers and reformats landscape video into perfect 9:16 shorts.</p>
                    </div>
                    <div className="absolute top-0 left-0 right-0 h-24 bg-gradient-to-b from-black/20 to-transparent" />
                    <div className="absolute bottom-0 left-0 right-0 h-40 bg-gradient-to-t from-black/40 to-transparent" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Queue + Clips */}
        <div className="mt-12 space-y-16">
          <Separator />
          
          <section>
            <div className="mb-6 flex items-center justify-between">
              <div>
                <h3 className="text-xl font-bold text-stone-900">Processing Queue</h3>
                <p className="text-sm text-stone-500 mt-1">Status of your recently uploaded videos</p>
              </div>
              <Button variant="outline" size="sm" onClick={handleRefresh} disabled={refreshing} className="bg-white">
                {refreshing ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Loader2 className="mr-2 h-4 w-4" />}
                Refresh
              </Button>
            </div>
            
            {uploadedFiles.length > 0 ? (
              <div className="overflow-hidden rounded-xl border border-stone-200 bg-white shadow-sm">
                <Table>
                  <TableHeader className="bg-stone-50">
                    <TableRow>
                      <TableHead className="font-medium">File Name</TableHead>
                      <TableHead className="font-medium hidden sm:table-cell">Uploaded</TableHead>
                      <TableHead className="font-medium">Status</TableHead>
                      <TableHead className="font-medium text-right">Clips</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {uploadedFiles.map((item) => (
                      <TableRow key={item.id} className="hover:bg-stone-50/50 border-stone-100">
                        <TableCell className="max-w-[150px] sm:max-w-xs truncate font-medium text-stone-900">{item.filename}</TableCell>
                        <TableCell className="text-stone-500 text-sm hidden sm:table-cell">{new Date(item.createdAt).toLocaleDateString()}</TableCell>
                        <TableCell>
                          {item.status === "queued" && <Badge variant="outline" className="bg-stone-50 text-stone-600 border-stone-200 shadow-none">Queued</Badge>}
                          {item.status === "processing" && (
                            <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200 shadow-none">
                              <Loader2 className="w-3 h-3 mr-1 animate-spin inline-block" />Processing
                            </Badge>
                          )}
                          {item.status === "processed" && <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200 shadow-none">Processed</Badge>}
                          {item.status === "failed" && <Badge variant="destructive" className="shadow-none text-xs">Failed</Badge>}
                        </TableCell>
                        <TableCell className="text-right">
                          {item.clipsCount > 0 ? (
                            <span className="inline-flex items-center justify-center bg-stone-100 text-stone-900 text-xs font-semibold px-2.5 py-0.5 rounded-full">
                              {item.clipsCount} clip{item.clipsCount !== 1 ? "s" : ""}
                            </span>
                          ) : (
                            <span className="text-stone-400 text-xs font-medium italic">None yet</span>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            ) : (
              <div className="py-12 text-center rounded-xl border border-dashed border-stone-200 bg-stone-50/50">
                <div className="w-12 h-12 bg-white rounded-full flex items-center justify-center mx-auto mb-3 shadow-sm border border-stone-100">
                  <List className="h-5 w-5 text-stone-400" />
                </div>
                <h3 className="text-sm font-medium text-stone-900">No videos in queue</h3>
                <p className="text-xs text-stone-500 mt-1 max-w-sm mx-auto">Upload a video above to start extracting viral moments.</p>
              </div>
            )}
          </section>

          <section className="pb-20">
            <div className="mb-6">
              <h3 className="text-xl font-bold text-stone-900">My Generated Clips</h3>
              <p className="text-sm text-stone-500 mt-1">Review, download, and share your viral moments</p>
            </div>
            <div className="bg-stone-50/30 rounded-2xl border border-stone-200/60 p-1 sm:p-6">
              <ClipDisplay clips={clips} />
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
