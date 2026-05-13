import { type MetadataRoute } from "next";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: "*",
        allow: ["/", "/auth/oauth/login"],
        disallow: ["/api/", "/admin/", "/tasks/", "/list/", "/settings/"],
      },
    ],
    sitemap: "https://clippedai.app/sitemap.xml",
    host: "https://clippedai.app",
  };
}
