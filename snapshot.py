#!/usr/bin/env python3
"""
Generate a single Markdown file containing the ENTIRE ClippedAI codebase.
Run after every change:  python snapshot.py

Output: CODEBASE.md — paste into any AI model for review.
"""

import os
import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(PROJECT_DIR, "CODEBASE.md")

# Files to include (relative to project root), in logical order
SOURCE_FILES = [
    "config.py",
    "modal_app.py",
    "run.py",
    "pipeline/__init__.py",
    "pipeline/ingest.py",
    "pipeline/transcribe.py",
    "pipeline/scene_detect.py",
    "pipeline/audio_analysis.py",
    "pipeline/clip_selector.py",
    "pipeline/reframe.py",
    "pipeline/captions.py",
    "pipeline/render.py",
    "requirements.txt",
    ".env.example",
    ".gitignore",
]

# Map extensions to markdown language hints
LANG_MAP = {
    ".py": "python",
    ".txt": "text",
    ".example": "bash",
    ".gitignore": "gitignore",
    ".md": "markdown",
}


def get_lang(filepath: str) -> str:
    basename = os.path.basename(filepath)
    if basename == ".gitignore":
        return "gitignore"
    _, ext = os.path.splitext(filepath)
    return LANG_MAP.get(ext, "text")


def generate_snapshot():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# ClippedAI — Full Codebase Snapshot")
    lines.append("")
    lines.append(f"> Generated: {timestamp}")
    lines.append(f"> Files: {len(SOURCE_FILES)}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Table of Contents")
    lines.append("")
    for f in SOURCE_FILES:
        anchor = f.replace("/", "").replace("_", "-").replace(".", "")
        lines.append(f"- [{f}](#{anchor})")
    lines.append("")
    lines.append("---")
    lines.append("")

    file_count = 0
    total_lines = 0

    for rel_path in SOURCE_FILES:
        abs_path = os.path.join(PROJECT_DIR, rel_path)
        if not os.path.exists(abs_path):
            lines.append(f"## `{rel_path}`")
            lines.append("")
            lines.append("**⚠️ FILE NOT FOUND**")
            lines.append("")
            lines.append("---")
            lines.append("")
            continue

        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        total_lines += line_count
        file_count += 1
        lang = get_lang(rel_path)

        lines.append(f"## `{rel_path}`")
        lines.append("")
        lines.append(f"*{line_count} lines*")
        lines.append("")
        lines.append(f"```{lang}")
        lines.append(content.rstrip())
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Summary at end
    lines.append(f"**Total: {file_count} files, {total_lines} lines**")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Snapshot saved: {OUTPUT_FILE}")
    print(f"   {file_count} files, {total_lines} lines")
    size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"   Size: {size_kb:.1f} KB")


if __name__ == "__main__":
    generate_snapshot()
