"""
Automated YouTube cookie refresh via Playwright Stealth.

Runs a headless Chromium session to generate fresh cookies + visitor_data
when yt-dlp encounters bot verification. Cached on Modal Volume with TTL.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)


def refresh_cookies(
    cookie_path: Path,
    account_email: Optional[str] = None,
    account_password: Optional[str] = None,
) -> Path:
    """
    Launch headless Chromium via Playwright Stealth, navigate to YouTube,
    and export cookies in Netscape format for yt-dlp.

    Returns path to the cookie file.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError(
            "Playwright not installed. Run: pip install playwright && "
            "playwright install chromium"
        )

    cookie_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )

        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Linux; Android 13; SM-G991B) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Mobile Safari/537.36"
            ),
            viewport={"width": 412, "height": 915},
            is_mobile=True,
        )

        page = context.new_page()

        try:
            # Navigate to YouTube
            logger.info("Navigating to YouTube for cookie refresh...")
            page.goto("https://www.youtube.com", wait_until="networkidle",
                      timeout=30000)
            time.sleep(3)  # Let JS settle

            # Accept cookies dialog if present
            try:
                accept_btn = page.locator(
                    "button:has-text('Accept'), "
                    "button:has-text('I agree'), "
                    "button:has-text('Accept all')"
                )
                if accept_btn.count() > 0:
                    accept_btn.first.click()
                    time.sleep(1)
            except Exception:
                pass  # No cookie dialog, continue

            # Extract cookies
            cookies = context.cookies()
            _write_netscape_cookies(cookies, cookie_path)

            logger.info(
                "Cookie refresh successful: %d cookies saved to %s",
                len(cookies), cookie_path,
            )

        except Exception as e:
            logger.error("Cookie refresh failed: %s", e)
            raise

        finally:
            browser.close()

    return cookie_path


def is_cookie_valid(cookie_path: Path) -> bool:
    """Check if cookie file exists and is within TTL."""
    if not cookie_path.exists():
        return False

    age = time.time() - cookie_path.stat().st_mtime
    if age > config.YT_COOKIE_TTL:
        logger.info("Cookies expired (%.0fs old, TTL=%ds)", age, config.YT_COOKIE_TTL)
        return False

    return True


def _write_netscape_cookies(cookies: list[dict], path: Path) -> None:
    """Write cookies in Netscape format that yt-dlp accepts."""
    lines = ["# Netscape HTTP Cookie File", ""]

    for c in cookies:
        domain = c.get("domain", "")
        if not domain.startswith("."):
            domain = "." + domain

        flag = "TRUE" if domain.startswith(".") else "FALSE"
        path_val = c.get("path", "/")
        secure = "TRUE" if c.get("secure", False) else "FALSE"
        expires = str(int(c.get("expires", 0)))
        name = c.get("name", "")
        value = c.get("value", "")

        lines.append(f"{domain}\t{flag}\t{path_val}\t{secure}\t{expires}\t{name}\t{value}")

    path.write_text("\n".join(lines))
