# Contributing

Issues and pull requests are welcome.

## Reporting a bad match

Open an issue with: the card (set code + collector number or Scryfall URL), a
photo of your rectified scan if possible, and the top distances your matcher
returned. Most mismatches trace back to the art-box crop on unusual layouts
(sagas, battles, split cards) — those reports are the most valuable.

## Changing the hash

`image_hash.py` is the reference implementation consumed by downstream apps.
Any change to the produced bits (crop box, planes, size, interpolation,
packing order) **must** bump `HASH_ALGO_VERSION` — the CI then rebuilds the
full index automatically, and consumers checking `meta.algo_version` fail
closed instead of mismatching silently. Include before/after matching margins
on real captures in the PR description.

## Build pipeline

`build_index.py` must stay incremental (a daily run fetches only missing
groups) and polite to Scryfall (keep the request delay). Test locally:

```
pip install httpx ijson opencv-python-headless numpy
python build_index.py            # uses/creates art_hashes.sqlite in place
```
