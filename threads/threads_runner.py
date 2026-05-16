import os
import re
import unicodedata
from typing import Iterable

import pandas as pd
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright


HEADLESS           = False
STORAGE_STATE_PATH = os.path.join(os.path.dirname(__file__), "threads_storage_state.json")
THREADS_BASE_URL   = "https://www.threads.com/"


def normalize_queries(queries: Iterable[str]) -> list[str]:
    normalized = []
    seen = set()

    for query in queries:
        clean_query = query.strip()
        if not clean_query:
            continue

        dedupe_key = clean_query.casefold()
        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        normalized.append(clean_query)

    return normalized


def normalize_text(value: str) -> str:
    value = value.strip().casefold()
    value = value.replace("đ", "d")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"\s+", " ", value)
    return value


def text_matches_queries(value: str, queries: list[str]) -> bool:
    normalized_value = normalize_text(value)
    return any(normalize_text(query) in normalized_value for query in queries)


def build_output_name(queries: list[str]) -> str:
    if len(queries) == 1:
        return slugify(queries[0])

    return "_".join(slugify(query) for query in queries)


def posts_to_dataframe(posts):
    return pd.DataFrame(
        posts,
        columns=["source_query", "post_url", "author_handle", "title", "text"],
    )


def comments_to_dataframe(comments):
    return pd.DataFrame(
        comments,
        columns=["post_url", "author_handle", "comment_text", "create_time"],
    )


def save_threads_csv(posts, comments, queries, output_dir=None):
    normalized_queries = normalize_queries(queries)
    output_name = build_output_name(normalized_queries)
    target_dir = output_dir or os.path.dirname(os.path.abspath(__file__))

    posts_filename = os.path.join(target_dir, f"{output_name}_posts.csv")
    comments_filename = os.path.join(target_dir, f"{output_name}_comments.csv")

    posts_to_dataframe(posts).to_csv(posts_filename, index=False, encoding="utf-8-sig")
    comments_to_dataframe(comments).to_csv(comments_filename, index=False, encoding="utf-8-sig")

    return {
        "posts_csv": posts_filename,
        "comments_csv": comments_filename,
    }


def slugify(value: str) -> str:
    value = value.casefold().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "threads"


def build_auth_cookies() -> list[dict]:
    session_id = os.getenv("THREADS_SESSIONID", "").strip()
    csrf_token = os.getenv("THREADS_CSRFTOKEN", "").strip()

    cookies = []
    if session_id:
        cookies.append(
            {
                "name": "sessionid",
                "value": session_id,
                "domain": ".threads.com",
                "path": "/",
                "httpOnly": True,
                "secure": True,
                "sameSite": "None",
            }
        )

    if csrf_token:
        cookies.append(
            {
                "name": "csrftoken",
                "value": csrf_token,
                "domain": ".threads.com",
                "path": "/",
                "httpOnly": False,
                "secure": True,
                "sameSite": "Lax",
            }
        )

    return cookies


async def click_first(page, selectors: list[str]) -> bool:
    for selector in selectors:
        locator = page.locator(selector).first
        try:
            if await locator.count() == 0:
                continue
            await locator.click(timeout=3000)
            return True
        except PlaywrightTimeoutError:
            continue
        except Exception:
            continue
    return False


async def fill_first(page, selectors: list[str], value: str) -> bool:
    for selector in selectors:
        locator = page.locator(selector).first
        try:
            if await locator.count() == 0:
                continue
            await locator.fill(value, timeout=3000)
            return True
        except PlaywrightTimeoutError:
            continue
        except Exception:
            continue
    return False


async def type_into_first(page, selectors: list[str], value: str) -> bool:
    for selector in selectors:
        locator = page.locator(selector).first
        try:
            if await locator.count() == 0:
                continue
            await locator.click(timeout=3000)
            await locator.press("Control+A", timeout=3000)
            await locator.press("Backspace", timeout=3000)
            await locator.type(value, delay=50, timeout=5000)
            return True
        except PlaywrightTimeoutError:
            continue
        except Exception:
            continue
    return False


async def prepare_threads_session(context):
    auth_cookies = build_auth_cookies()
    if auth_cookies:
        await context.add_cookies(auth_cookies)
        print("Da nap session cookie tu bien moi truong.")

    page = await context.new_page()
    await page.goto(THREADS_BASE_URL, wait_until="domcontentloaded")

    if "login" in page.url:
        print("Cookie hien tai chua du. Hay dang nhap tren cua so browser da mo.")
        await page.pause()
        await context.storage_state(path=STORAGE_STATE_PATH)
        print(f"Da luu storage state vao: {STORAGE_STATE_PATH}")

    return page


async def open_search(page, query: str):
    await page.goto(f"{THREADS_BASE_URL}search", wait_until="domcontentloaded")
    await page.wait_for_timeout(2500)

    search_ready = await fill_first(
        page,
        [
            'input[placeholder*="Search"]',
            'input[placeholder*="search"]',
            'input[aria-label*="Search"]',
            'input[type="search"]',
            'input[type="text"]',
        ],
        query,
    )

    if not search_ready:
        search_ready = await fill_first(
            page,
            [
                'textarea[placeholder*="Search"]',
                'textarea[aria-label*="Search"]',
            ],
            query,
        )

    if not search_ready:
        search_ready = await type_into_first(
            page,
            [
                '[contenteditable="true"][role="textbox"]',
                '[contenteditable="true"]',
                '[role="searchbox"]',
            ],
            query,
        )

    if not search_ready:
        await click_first(
            page,
            [
                'a[href="/search"]',
                'svg[aria-label="Search"]',
                '[role="link"][href="/search"]',
                '[aria-label="Search"]',
            ],
        )
        await page.wait_for_timeout(1500)
        search_ready = await fill_first(
            page,
            [
                'input[placeholder*="Search"]',
                'input[placeholder*="search"]',
                'input[aria-label*="Search"]',
                'input[type="search"]',
                'input[type="text"]',
                'textarea[placeholder*="Search"]',
                'textarea[aria-label*="Search"]',
            ],
            query,
        )

    if not search_ready:
        search_ready = await type_into_first(
            page,
            [
                '[contenteditable="true"][role="textbox"]',
                '[contenteditable="true"]',
                '[role="searchbox"]',
            ],
            query,
        )

    if not search_ready:
        debug_path = os.path.join(os.path.dirname(__file__), "threads_search_debug.png")
        await page.screenshot(path=debug_path, full_page=True)
        raise RuntimeError(f"Khong tim thay o search cua Threads. Da luu screenshot tai {debug_path}")

    await page.keyboard.press("Enter")
    await page.wait_for_timeout(3000)


async def collect_post_links(page, post_limit: int) -> list[str]:
    collected = []
    seen = set()

    for _ in range(10):
        links = await page.locator('a[href*="/post/"]').evaluate_all(
            """elements => elements.map(el => el.href).filter(Boolean)"""
        )

        for link in links:
            clean_link = link.removesuffix("/media")
            if clean_link in seen:
                continue
            seen.add(clean_link)
            collected.append(clean_link)
            if len(collected) >= post_limit:
                return collected

        await page.mouse.wheel(0, 4000)
        await page.wait_for_timeout(2000)

    return collected[:post_limit]


async def collect_candidate_posts(context, post_links: list[str], queries: list[str], post_limit: int) -> list[tuple[str, str]]:
    candidates = []

    for post_url in post_links:
        page = await context.new_page()
        try:
            await page.goto(post_url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(1500)

            title = await page.title()
            first_article = page.locator("article").first
            article_text = " ".join(await first_article.all_inner_texts()).strip() if await first_article.count() else ""
            combined_text = f"{title}\n{article_text}"

            if text_matches_queries(combined_text, queries):
                candidates.append((post_url, combined_text))
                if len(candidates) >= post_limit:
                    return candidates
        except Exception as exc:
            print(f"Khong danh gia duoc post {post_url}: {exc}")
        finally:
            await page.close()

    return candidates


async def extract_post_data(context, post_url: str, source_query: str) -> dict | None:
    page = await context.new_page()
    try:
        await page.goto(post_url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2000)

        page_title = await page.title()
        article_text = " ".join(await page.locator("article").first.all_inner_texts()).strip()
        meta_description = await page.locator('meta[name="description"]').get_attribute("content")

        author_handle = None
        match = re.search(r"threads\.com\/@([^\/]+)\/post\/", post_url)
        if match:
            author_handle = match.group(1)

        return {
            "source_query": source_query,
            "post_url": post_url,
            "author_handle": author_handle,
            "title": page_title,
            "text": article_text or meta_description or "",
        }
    except Exception as exc:
        print(f"Khong doc duoc post {post_url}: {exc}")
        return None
    finally:
        await page.close()


async def extract_comments_from_post(context, post_url: str, comments_per_post: int) -> list[dict]:
    page = await context.new_page()
    comments_data = []
    seen_comment_keys = set()

    try:
        await page.goto(post_url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2500)

        for _ in range(8):
            containers = page.locator('article, div[data-pressable-container="true"]')
            container_count = await containers.count()

            for index in range(container_count):
                container = containers.nth(index)
                article_text = " ".join(await container.all_inner_texts()).strip()
                if not article_text:
                    continue

                author_link = container.locator('a[href^="/@"]').first
                time_node = container.locator("time").first
                if await author_link.count() == 0 or await time_node.count() == 0:
                    continue

                href = await author_link.get_attribute("href")
                timestamp = await time_node.get_attribute("datetime") or ""
                if not href or not timestamp:
                    continue

                author_handle = href.split("/@")[-1].split("/")[0]

                # Bo qua post goc, chi lay replies nam duoi bai viet.
                if post_url.rstrip("/") in page.url.rstrip("/") and index == 0:
                    continue

                comment_key = (author_handle, timestamp, article_text)
                if comment_key in seen_comment_keys:
                    continue

                seen_comment_keys.add(comment_key)
                comments_data.append(
                    {
                        "post_url": post_url,
                        "author_handle": author_handle,
                        "comment_text": article_text,
                        "create_time": timestamp,
                    }
                )

                if len(comments_data) >= comments_per_post:
                    return comments_data

            await page.mouse.wheel(0, 3000)
            await page.wait_for_timeout(1500)

        if not comments_data:
            debug_path = os.path.join(os.path.dirname(__file__), "threads_comments_debug.png")
            await page.screenshot(path=debug_path, full_page=True)
            print(f"Khong tim thay comments cho {post_url}. Da luu screenshot tai {debug_path}")

        return comments_data
    except Exception as exc:
        print(f"Khong doc duoc comments cho post {post_url}: {exc}")
        return comments_data
    finally:
        await page.close()


async def get_threads_data(queries, post_limit_per_query=2, comments_per_post=20):
    queries = normalize_queries(queries)
    posts_data = []
    comments_data = []
    seen_post_urls = set()

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=HEADLESS)
        context_kwargs = {}
        if os.path.exists(STORAGE_STATE_PATH):
            context_kwargs["storage_state"] = STORAGE_STATE_PATH
        context = await browser.new_context(**context_kwargs)

        page = await prepare_threads_session(context)

        for query in queries:
            print(f"\nDang tim post Threads voi tu khoa: {query}")
            await open_search(page, query)
            post_links = await collect_post_links(page, post_limit_per_query)
            candidate_posts = await collect_candidate_posts(context, post_links, queries, post_limit_per_query)
            print(f"Tim thay {len(candidate_posts)} post phu hop sau khi chuan hoa query.")

            for post_url, _ in candidate_posts:
                if post_url in seen_post_urls:
                    continue

                seen_post_urls.add(post_url)
                post_data = await extract_post_data(context, post_url, query)
                if not post_data:
                    continue

                posts_data.append(post_data)

                if comments_per_post > 0:
                    post_comments = await extract_comments_from_post(
                        context,
                        post_url,
                        comments_per_post,
                    )
                    comments_data.extend(post_comments)

        await page.close()
        await context.close()
        await browser.close()

    return posts_data, comments_data


# ─── DataFrame builders ───────────────────────────────────────────────────────

def posts_to_dataframe(posts: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(posts)


def comments_to_dataframe(comments: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(comments)

# ─── Synchronous wrapper ──────────────────────────────────────────────────────

import threading
import asyncio
import sys

def run_async_in_thread(coro_func, *args, **kwargs):
    result = None
    exception = None

    def thread_target():
        nonlocal result, exception
        try:
            if sys.platform.startswith('win'):
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro_func(*args, **kwargs))
            loop.close()
        except Exception as e:
            exception = e

    t = threading.Thread(target=thread_target)
    t.start()
    t.join()

    if exception:
        raise exception
    return result

def get_threads_data_sync(queries, post_limit_per_query=2, comments_per_post=20):
    return run_async_in_thread(get_threads_data, queries, post_limit_per_query, comments_per_post)

