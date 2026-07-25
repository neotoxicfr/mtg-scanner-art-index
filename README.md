# mtg-scanner-art-index

**A prebuilt, daily-updated perceptual-hash index of every Magic: The
Gathering card illustration — for building card scanner apps.**

Card recognition apps all solve the same cold-start problem: downloading tens
of thousands of card images and fingerprinting them before the first scan can
work. This repository does that work once, in public, and publishes the result
as a small file anyone can consume.

- **60,000+ entries**, one per `(illustration_id, frame)` group — every unique
  artwork in every frame style it was printed in (see the release title for
  the exact current count)
- **1024-bit hybrid perceptual hash** per entry (grayscale + B/G/R channel
  dHashes of the art region)
- Published daily as the [`latest` release](../../releases/latest) asset
  **`art_hashes.sqlite`** (~10 MB)
- Built incrementally from Scryfall's `default_cards` bulk: a normal day
  fetches 0–50 new images, a set release a few hundred
- No card images or text are redistributed — only derived numeric
  fingerprints and Scryfall identifiers

## Why (illustration, frame) groups instead of one hash per card?

An image hash can only distinguish what is *visible in pixels*. Printings that
share an illustration and a frame style are visually identical apart from tiny
regions (set symbol, collector line) that a 1024-bit fingerprint cannot
reliably separate — and language variants share everything but the text.

Hashing one representative image per `(illustration_id, frame)` group makes
the semantics honest: **a match returns the set of compatible printings.**
Your app then narrows within the group using signals a hash cannot see —
collector-line OCR, language of the printed title, foil glyphs, The List's
planeswalker stamp — or simply asks the user to pick among a handful of
versions.

## Data format

`art_hashes.sqlite` contains two tables:

```sql
CREATE TABLE art_hashes (
    illustration_id TEXT NOT NULL,   -- Scryfall illustration_id
    frame           TEXT NOT NULL,   -- Scryfall frame: 1993|1997|2003|2015|future
    hash            BLOB NOT NULL,   -- 128 bytes, see spec below
    PRIMARY KEY (illustration_id, frame)
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
-- meta keys: algo_version, groups, bulk_updated_at
```

Map a matched group back to concrete printings with Scryfall's own data:
every card object carries `illustration_id` and `frame`.

## Hash specification (algo_version 1)

The reference implementation is [`image_hash.py`](image_hash.py) (pure
OpenCV + NumPy, no other dependencies). To interoperate you must reproduce it
exactly — check `meta.algo_version` against `HASH_ALGO_VERSION` before using
an index.

1. **Art crop**: relative box `y 11%–54%`, `x 9%–91%` of the full card image.
   The edges intentionally include a sliver of frame — that is what separates
   the same artwork printed in different frame styles.
2. **Four 256-bit dHash planes**, concatenated in order: grayscale
   (`cv2.COLOR_BGR2GRAY`), then B, G, R channels. Each plane: resize the crop
   to 17×16 (`INTER_AREA`), compare horizontal neighbours
   (`pixel[x+1] > pixel[x]`), pack row-major with `np.packbits`.
3. **Total**: 128 bytes per entry.

**Matching a camera scan**: rectify the card to a portrait rectangle first,
then hash it at several vertical offsets of the art box (the reference uses
`0, −2%, +2%, +4%`) and keep the minimum Hamming distance per entry — this
absorbs imperfect framing. On a 59k index, a full match is a few milliseconds
with hardware popcount (see `example/match_example.py`).

Calibration reference (4K webcam captures vs this index): true group at
Hamming 180–340 of 1024, best impostor 46–150 bits further, random ≈ 512.

## Quick start

```python
# pip install opencv-python numpy
python example/match_example.py path/to/rectified_card.png
```

See [`example/match_example.py`](example/match_example.py) — ~40 lines,
downloads nothing, depends only on the release asset.

## What the hash cannot tell you

By design, a group match does **not** distinguish:

- **language** — Scryfall localizes images, but art and frame are identical
- **foil / etched** — same printed image
- **The List reprints** — identical to the original except a small
  planeswalker stamp bottom-left
- **printings sharing art and frame** — e.g. a card reprinted unchanged

These are features, not bugs: they keep the index small and the semantics
truthful. Resolve them downstream with OCR or user choice.

Known limitations:

- **Double-faced cards**: only the front face is indexed; back-face scans
  need another identification path.
- Query offsets cover vertical framing drift only; strong horizontal
  misframing degrades distances.

## Build pipeline

[`build_index.py`](build_index.py) runs daily via GitHub Actions:
download the previous release → diff Scryfall's `default_cards` bulk for
missing illustrations → fetch only those images (rate-limited, `small` size)
→ hash → publish. Only paper printings are indexed (`games` must contain
`paper`): digital-only arts (rebalanced Alchemy, Arena/MTGO exclusives)
cannot be scanned. A change to the hash parameters (crop box or that filter)
changes `crop_signature` and triggers a full rebuild automatically.

[`extra_groups.json`](extra_groups.json) pins the localized art variants that
share an English printing's collector number and therefore never appear in
`default_cards`. Only variants whose illustration is *also* used by an
English or French printing are pinned — an art exclusive to a language a
consumer never imports would index a hash no card can ever reference (git
history keeps the pruned ones if that ever changes).

## Legal

Code is MIT-licensed. The published data consists of derived numeric
fingerprints and Scryfall identifiers; no Wizards of the Coast imagery or
text is redistributed. Card data courtesy of [Scryfall](https://scryfall.com);
Magic: The Gathering is © Wizards of the Coast. This project is unofficial
fan content permitted under the Fan Content Policy and is not approved or
endorsed by Wizards.
