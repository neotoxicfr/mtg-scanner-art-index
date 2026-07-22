# mtg-scanner-art-index

Prebuilt perceptual-hash index of Magic: The Gathering card illustrations,
consumed by [mtgscan](https://github.com/neotoxicfr/mtgscan).

- One 1024-bit hybrid dHash (gray + BGR channels) per `(illustration_id, frame)`
  group, computed by `mtgscan.vision.image_hash` — the app repository is the
  single source of truth for the hash function (`HASH_ALGO_VERSION`).
- Published daily as the `latest` release asset `art_hashes.sqlite`
  (`art_hashes` table + `meta` table with `algo_version`, `groups`,
  `bulk_updated_at`).
- Incremental: each run diffs Scryfall's `default_cards` bulk against the
  previous release and fetches only missing card images (rate-limited).
  An algo version change triggers a full rebuild automatically.

No card images or text are published — only derived numeric fingerprints and
Scryfall identifiers.

## Setup

1. Repository secret `APP_REPO_TOKEN`: fine-grained PAT with read access to
   the app repository (needed while it is private, for `pip install git+...`).
2. Seed the first release (from a machine holding a built index):

   ```
   python tools/export_art_index.py            # in the app repository
   gh release create latest art_hashes.sqlite --repo <owner>/mtg-scanner-art-index
   ```

3. The daily workflow keeps it updated; `workflow_dispatch` allows manual runs.
