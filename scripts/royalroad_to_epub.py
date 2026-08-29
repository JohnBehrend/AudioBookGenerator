#!/usr/bin/env python3
"""Download a Royal Road web novel and build an EPUB.

Usage:
    python3 scripts/royalroad_to_epub.py <fiction_url> [--out OUT.epub] [--max-chapters N]
"""

import argparse
import re
import sys
import time

import requests
from bs4 import BeautifulSoup
from ebooklib import epub

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
DELAY = 1.0

# Titles that are author's notes / announcements rather than story chapters.
NOTE_TITLE_RE = re.compile(
    r"^(update|announcement|author'?s?\s*note|author\s*note|a\s*\.?\s*n\.?\s*|notice|"
    r"status\s*update|side\s*note)\b",
    re.IGNORECASE,
)


def fetch(url: str) -> BeautifulSoup:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser")
        except requests.RequestException as e:
            print(f"  retry ({attempt + 1}/3): {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}")


def get_chapter_links(fiction_url: str):
    soup = fetch(fiction_url)
    rows = soup.select("tr.chapter-row")
    links = []
    for row in rows:
        a = row.find("a")
        if a and a.get("href"):
            title = a.get_text(strip=True)
            href = a["href"]
            if not href.startswith("http"):
                href = "https://www.royalroad.com" + href
            links.append((href, title))
    return links


def is_author_note(href: str, title: str) -> bool:
    """Return True if a chapter is an author's note / update, not story content."""
    if NOTE_TITLE_RE.match(title):
        return True
    if href.rstrip("/").endswith(("/update", "/announcement")):
        return True
    return False


def get_chapter_content(chapter_url: str, title: str):
    soup = fetch(chapter_url)
    div = soup.select_one(".chapter-content")
    if div is None:
        raise RuntimeError(f"No .chapter-content in {chapter_url}")
    paragraphs = [p.get_text("\n", strip=False).strip() for p in div.find_all("p")]
    paragraphs = [p for p in paragraphs if p]
    if not paragraphs:
        text = div.get_text("\n", strip=True)
        paragraphs = [p for p in text.split("\n") if p.strip()]
    return paragraphs


def build_epub(links, title, author, out_path, progress=False):
    book = epub.EpubBook()
    book.set_identifier(f"royalroad-{re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')}")
    book.set_title(title)
    book.set_language("en")
    book.add_author(author)

    chapters = []
    total = len(links)
    for i, (href, chap_title) in enumerate(links, start=1):
        if progress:
            print(f"[{i}/{total}] {chap_title}")
        paragraphs = get_chapter_content(href, chap_title)
        body = "".join(f"<p>{p}</p>" for p in paragraphs)
        c = epub.EpubHtml(
            title=chap_title,
            file_name=f"chapter_{i:04d}.xhtml",
            lang="en",
        )
        c.content = f"<h1>{chap_title}</h1>{body}"
        book.add_item(c)
        chapters.append(c)
        time.sleep(DELAY)

    book.toc = tuple(chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + chapters
    epub.write_epub(out_path, book, {})
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Download a Royal Road novel into an EPUB.")
    ap.add_argument("fiction_url", help="Story URL, e.g. https://www.royalroad.com/fiction/131957/the-breakwall-paladin")
    ap.add_argument("--out", default=None, help="Output EPUB path (default: <slug>.epub)")
    ap.add_argument("--max-chapters", type=int, default=None, help="Limit number of chapters")
    args = ap.parse_args()

    soup = fetch(args.fiction_url)
    title_el = soup.select_one("div.fic-header h1")
    title = title_el.get_text(strip=True) if title_el else "Untitled"
    author_el = soup.select_one("div.fic-header h4 a")
    author = author_el.get_text(strip=True) if author_el else "Unknown"
    print(f"Title: {title}\nAuthor: {author}")

    links = get_chapter_links(args.fiction_url)
    if args.max_chapters:
        links = links[: args.max_chapters]
    story = [l for l in links if not is_author_note(l[0], l[1])]
    skipped = len(links) - len(story)
    print(f"Chapters: {len(links)} ({skipped} author note(s) skipped)")

    if args.out:
        out_path = args.out
    else:
        slug = re.sub(r"^/fiction/\d+/", "", re.match(r".*?(/fiction/[^\s]+)", args.fiction_url).group(1))
        slug = slug.rstrip("/")
        out_path = f"{slug.split('/')[-1]}.epub"

    build_epub(story, title, author, out_path, progress=True)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
