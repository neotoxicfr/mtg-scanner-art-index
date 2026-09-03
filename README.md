# mtg-scanner-art-index

**A prebuilt, daily-updated perceptual-hash index of every Magic: The
Gathering card illustration — for building card scanner apps.**

Card recognition apps all solve the same cold-start problem: downloading tens
of thousands of card images and fingerprinting them before the first scan can
work. This repository does that work once, in public, and publishes the result
as a small file anyone can consume.

- **~50,000 entries**, one per Scryfall `illustration_id` — every unique
  artwork (see the release title for the exact current count)
- **1024-bit hybrid perceptual hash** per entry (grayscale + B/G/R channel
  dHashes of the interior art region), plus an optional **DINOv2 embedding**
- Published daily as the [`latest` release](../../releases/latest) asset
  **`art_hashes.sqlite`** (~37 MB)
- Built incrementally from Scryfall's `default_cards` bulk: a normal day
  fetches 0–50 new images, a set release a few hundred
- No card images or text are redistributed — only derived numeric
  fingerprints and Scryfall identifiers

## Why one hash per illustration (not per printing)?

An image hash can only distinguish what is *visible in pixels*. The art crop
is taken **interior to the illustration**, excluding the frame entirely, so
every printing of the same artwork — across frame styles, languages, foils,
and reprints — hashes to the **same** fingerprint. Set symbol and collector
line sit outside the crop and never reach the hash.

This makes the semantics honest: **a match returns the set of compatible
printings.** Your app then narrows within the group using signals a hash
cannot see — collector-line OCR, language of the printed title, foil glyphs,
The List's planeswalker stamp — or simply asks the user to pick among a
handful of versions.

## Data format

`art_hashes.sqlite` contains three tables:

```sql
CREATE TABLE art_hashes (
    illustration_id TEXT NOT NULL PRIMARY KEY,  -- Scryfall illustration_id
    hash            BLOB NOT NULL               -- 128 bytes, see spec below
);
CREATE TABLE art_embeddings (
    illustration_id TEXT NOT NULL PRIMARY KEY,  -- joins art_hashes
    embedding       BLOB NOT NULL               -- 384 int8, DINOv2-small (see embed.py)
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
-- meta keys: algo_version, embed_version, crop_signature, groups,
--            bulk_updated_at, image_updated_through, manifest_resume,
--            retry_images (the last two are build bookkeeping, usually empty)
```

Map a matched `illustration_id` back to concrete printings with Scryfall's own
data: every card object carries `illustration_id`.

The **`art_hashes`** table (the dHash) is the load-bearing channel and all you
need to match. **`art_embeddings`** is an optional second signal for
tie-breaking near-collisions; use it only if your embedder reproduces
`embed.py` exactly and `meta.embed_version` matches your copy — otherwise
ignore it and rely on the dHash.

## Hash specification (algo_version 1)

The reference implementation is [`image_hash.py`](image_hash.py) (pure
OpenCV + NumPy, no other dependencies). To interoperate you must reproduce it
exactly — check `meta.algo_version` against `HASH_ALGO_VERSION` before using
an index.

1. **Art crop**: relative box `y 16%–50%`, `x 14%–86%` of the full card image
   (`ART_TOP/BOTTOM/LEFT/RIGHT` in `image_hash.py`). The box is **interior to
   the art** — no frame — which is what makes every frame style of one
   illustration hash identically.
2. **Four 256-bit dHash planes**, concatenated in order: grayscale
   (`cv2.COLOR_BGR2GRAY`), then B, G, R channels. Each plane: resize the crop
   to 17×16 (`INTER_AREA`), compare horizontal neighbours
   (`pixel[x+1] > pixel[x]`), pack row-major with `np.packbits`.
3. **Total**: 128 bytes (1024 bits) per entry. Input must be a 3-channel BGR
   image; consumers need `numpy >= 2.0` (for `np.bitwise_count`).

**Matching a camera scan**: rectify the card to a portrait rectangle first,
then hash it at the several art-box offsets the reference uses
(`_QUERY_OFFSETS`: vertical `0, −2%, +2%, +4%` and horizontal `±1.1%`, six in
all) and keep the minimum Hamming distance per entry — this absorbs imperfect
framing. On a ~50k index, a full match is a few milliseconds with hardware
popcount (see `example/match_example.py`).

Calibration reference (4K webcam captures vs this index): true group at
Hamming 180–340 of 1024, best impostor 46–150 bits further, random ≈ 512.

## Quick start

```bash
pip install opencv-python-headless "numpy>=2.0"
python example/match_example.py path/to/rectified_card.png
```

See [`example/match_example.py`](example/match_example.py) — downloads nothing,
depends only on the release asset.

## What the hash cannot tell you

By design, an `illustration_id` match does **not** distinguish:

- **language** — Scryfall localizes images, but the art is identical
- **foil / etched** — same printed image
- **The List reprints** — identical to the original except a small
  planeswalker stamp bottom-left
- **frame style / printings sharing art** — the crop is interior, so all hash
  the same

These are features, not bugs: they keep the index small and the semantics
truthful. Resolve them downstream with OCR or user choice.

Known limitations:

- **Double-faced cards**: only the front face is indexed; back-face scans
  need another identification path.
- Query offsets cover mostly vertical framing drift; strong horizontal
  misframing degrades distances.

## Build pipeline

[`build_index.py`](build_index.py) runs daily via GitHub Actions:
download the previous release → diff Scryfall's `default_cards` bulk for
missing illustrations → fetch only those images (rate-limited, `small` size)
→ hash and embed → publish. Only paper printings are indexed (`games` must
contain `paper`): digital-only arts (rebalanced Alchemy, Arena/MTGO
exclusives) cannot be scanned. A change to the hash parameters (crop box or
that filter) changes `crop_signature` and triggers a full rebuild
automatically.

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
