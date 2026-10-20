# Candy shimeji

Scripts that turn `images/candy_stare.jpg` into the homepage shimeji's images in `images/shimeji/`.
The widget itself (markup, CSS, JS) is `_includes/candy-shimeji.html`, included at the end of
`_pages/about.html`. Jekyll ignores this folder, so none of it is published.

## Rebuild

```sh
python3 -m venv /tmp/shimeji-venv
/tmp/shimeji-venv/bin/pip install -r _shimeji/requirements.txt
PYTHON=/tmp/shimeji-venv/bin/python _shimeji/build.sh
```

The first run downloads the BiRefNet background-removal model (about 1 GB, to `~/.u2net`; set `U2NET_HOME`
to put it elsewhere). The cutout is cached in the work dir, so later runs only redo the fast steps.
Check `preview_eyes.png` and `preview_tail.gif` in the work dir before committing.

## Pipeline

| step | script | output |
| --- | --- | --- |
| 1 | `cutout.py` | the cat cut out of the photo (rembg, BiRefNet) |
| 2 | `build_master.py` | cat with cleaned-up fur edges, plus the floor shadow measured from the photo (left side and under the chest; the right side's darkness comes from the tunnel out of frame, so only a faint contact shadow is added there) |
| 3 | `eyes.py` | 9 gaze frames: looking at you, n, ne, e, se, s, sw, w, nw |
| 4 | `tail_spine.py` | the tail's centreline, root to tip |
| 5 | `tail.py` | 6 tail-flick keyframes: rest, up1, up2 (peak), down1, down2, over (small overshoot) |
| 6 | `build_assets.py` | `candy.webp`, `candy-eyes.webp`, `candy-tail.webp` and `candy-shimeji.json` (CSS geometry) |

## What to tweak where

- Eye geometry (eye openings, pupils, catchlights) and travel: constants at the top of `eyes.py`.
- Tail flick amplitude and shape: `KEYFRAMES` in `tail.py`. Its timing is `FLICK` in the include
  (and the matching `seq` in `preview.py`).
- Display size, close button, gaze dead zone: the include.
- If a rebuild changes the sprite boxes or the grab outline (`eye_patch_pct`, `tail_patch_pct`,
  `grab_polygon` in `candy-shimeji.json`), copy the new numbers into the include's CSS.
