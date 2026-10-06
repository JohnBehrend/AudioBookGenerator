# .jbab container format

A `.jbab` file is a single-file audiobook container produced by
`scripts/jbab_pack.py` and consumed by the reader app. It bundles chapter
audio, per-word karaoke timings, and reading text into one archive.

## Layout

A `.jbab` is a zip archive containing a `manifest.json` at the root plus the
chapter files it references (paths in the manifest are relative to the
archive root):

```
manifest.json
chapter_00.mp3
chapter_00.txt      (optional: reading text)
chapter_00.words.json (optional: word timings)
...
```

## manifest.json

```json
{
  "v": 1,
  "title": "Book Title",
  "author": "Author Name",
  "series": "Series Name",
  "series_num": 3,
  "chapters": [
    {"file": "chapter_00.mp3", "title": "chapter_00"}
  ]
}
```

| field        | type   | required | notes |
|--------------|--------|----------|-------|
| `v`          | int    | yes      | format version; currently `1` |
| `title`      | string | yes      | book title |
| `author`     | string | yes      | author (may be empty) |
| `series`     | string | no       | series name; empty/absent = standalone |
| `series_num` | int    | no       | book number within the series; `0` = unset |
| `chapters`   | list   | yes      | ordered chapter list |

### Chapter entries

| field   | type   | notes |
|---------|--------|-------|
| `file`  | string | audio file path relative to the archive root |
| `title` | string | chapter display title (file stem by default) |

## Sidecar files (per chapter)

- `<stem>.txt` — reading text, one `Line N: <text>` per TTS line. Used as the
  karaoke display text when present.
- `<stem>.words.json` — word timings from the ASR alignment pass:
  `{"v": 1, "words": [[start_ms, end_ms], ...]}` in reading-text order.

When `words.json` is absent, readers should fall back to even word timing.

## Versioning rule

New fields are **additive only**: readers must ignore unknown manifest keys.
Bump `v` only for breaking changes (renames, semantic changes to existing
keys).

## Building

```sh
uv run python scripts/jbab_pack.py <book-dir> -o out.jbab --title "..." \
    --author "..." --series "..." --series-num 3

uv run python scripts/jbab_batch.py --root voice_test --out voice_test/jbab_out
```

`jbab_batch.py` reads book metadata (dir -> title/series/num) from
`voice_test/books.json` (gitignored — book data stays out of the repo):

```json
{"<dir-name>": {"title": "Book Title", "series": "Series", "num": 3}}
```
