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

No secrets required: the hash function is vendored (`image_hash.py`, kept in
sync with the app; `HASH_ALGO_VERSION` guards divergence) and releases are
published with the workflow's automatic token.

1. Seed the first release (from a machine holding a built index), or simply
   run the workflow manually (`workflow_dispatch`) and let it build from
   scratch (~2-3h).
2. The daily workflow keeps it updated incrementally.

While this repository is private, the app cannot fetch releases without a
token and keeps using its locally built index; once public, the app updates
from here automatically and unauthenticated.
