"""Pack the frames into web assets.

usage: build_assets.py <eye_frames_dir> <tail_frames_dir> <master_cat.png> <outdir> [width_px]
Writes
  candy.webp       the cat at rest, centre gaze; inside the tail patch only pixels that are opaque and the
                   same in every tail frame are kept (the rest comes from the tail sprite, so nothing is
                   drawn twice and no seam opens up at fractional scales)
  candy-eyes.webp  eye patch for every gaze, side by side (feathered edge; it sits on opaque fur)
  candy-tail.webp  tail patch for every tail keyframe, side by side
  candy-shimeji.json  geometry for the CSS (patch boxes as % of the image, grab polygon)
"""
import json
import sys

import cv2
import numpy as np
from PIL import Image

edir, tdir, cat_path, out = sys.argv[1:5]
W = int(sys.argv[5]) if len(sys.argv) > 5 else 360      # 2x of the 180 CSS px display width
EYES = ["c", "n", "ne", "e", "se", "s", "sw", "w", "nw"]
TAIL = ["rest", "up1", "up2", "down1", "down2", "over"]
emeta = json.load(open(f"{edir}/frames_meta.json"))


def shrink(img, size):
    # premultiplied resize, so transparent pixels don't bleed a dark fringe into the fur edge
    return img.convert("RGBa").resize(size, Image.LANCZOS).convert("RGBA")


def even_box(x0, y0, x1, y1, pad, limit):
    x0, y0 = max(0, int(np.floor(x0)) - pad), max(0, int(np.floor(y0)) - pad)
    x1, y1 = min(limit[0], int(np.ceil(x1)) + pad), min(limit[1], int(np.ceil(y1)) + pad)
    return x0 - x0 % 2, y0 - y0 % 2, x1 + x1 % 2, y1 + y1 % 2


first = Image.open(f"{tdir}/tail_rest.png")
H = round(first.height * W / first.width)
H += H % 2
sx, sy = W / first.width, H / first.height
eyes = {d: np.asarray(shrink(Image.open(f"{edir}/frame_{d}.png"), (W, H))) for d in EYES}
tail = {t: np.asarray(shrink(Image.open(f"{tdir}/tail_{t}.png"), (W, H))) for t in TAIL}
base = tail["rest"].copy()

# --- eye patch (opaque fur all around it, so a feathered overlay is safe)
ex0, ey0, ex1, ey1 = emeta["eye_region_master"]
epx0, epy0, epx1, epy1 = even_box(ex0 * sx, ey0 * sy, ex1 * sx, ey1 * sy, 2, (W, H))
epw, eph = epx1 - epx0, epy1 - epy0
for d in EYES:
    diff = np.abs(eyes[d].astype(int) - eyes["c"].astype(int)).max(-1) > 1
    ys, xs = np.nonzero(diff)
    if diff.any():
        assert xs.min() >= epx0 + 2 and xs.max() < epx1 - 2 and ys.min() >= epy0 + 2 and ys.max() < epy1 - 2, d
yy, xx = np.mgrid[:eph, :epw]
feather = np.clip((np.minimum.reduce([xx, yy, epw - 1 - xx, eph - 1 - yy]) + 0.5) / 2.0, 0, 1)
eye_sprite = np.zeros((eph, epw * len(EYES), 4), np.float32)
for i, d in enumerate(EYES):
    patch = eyes[d][epy0:epy1, epx0:epx1].astype(np.float32)
    patch[..., 3] *= feather
    eye_sprite[:, i * epw:(i + 1) * epw] = patch

# --- tail patch: everything that differs between tail frames, plus a margin
changed = np.zeros((H, W), bool)
for t in TAIL[1:]:
    changed |= np.abs(tail[t].astype(int) - tail["rest"].astype(int)).max(-1) > 1
ys, xs = np.nonzero(changed)
tpx0, tpy0, tpx1, tpy1 = even_box(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 4, (W, H))
tpw, tph = tpx1 - tpx0, tpy1 - tpy0
stack = np.stack([tail[t][tpy0:tpy1, tpx0:tpx1].astype(int) for t in TAIL])
keep = (stack[..., 3] == 255).all(0) & ((stack.max(0) - stack.min(0)).max(-1) <= 1)
region = base[tpy0:tpy1, tpx0:tpx1]
region[~keep] = 0
tail_sprite = np.concatenate([tail[t][tpy0:tpy1, tpx0:tpx1] for t in TAIL], 1)
# the base's eye region must still match the centre-gaze eye frame
assert np.abs(base[epy0:epy1, epx0:epx1].astype(int) - eyes["c"][epy0:epy1, epx0:epx1].astype(int)).max() <= 2

# --- grab area: the cat's silhouette (no shadow), simplified
cat_a = np.asarray(shrink(Image.open(cat_path), (W, H)))[..., 3]
mask = (cat_a > 128).astype(np.uint8)
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = max(cnts, key=cv2.contourArea)
poly = cv2.approxPolyDP(cnt, 1.6, True)[:, 0, :]
poly_css = ", ".join(f"{100 * x / W:.1f}% {100 * y / H:.1f}%" for x, y in poly)

Image.fromarray(base, "RGBA").save(f"{out}/candy.webp", quality=86, method=6)
Image.fromarray(eye_sprite.round().astype(np.uint8), "RGBA").save(f"{out}/candy-eyes.webp", quality=92, method=6)
Image.fromarray(tail_sprite, "RGBA").save(f"{out}/candy-tail.webp", quality=88, method=6)
Image.fromarray(base, "RGBA").save(f"{out}/candy_check.png")
Image.fromarray(eye_sprite.round().astype(np.uint8), "RGBA").save(f"{out}/candy-eyes_check.png")
Image.fromarray(tail_sprite, "RGBA").save(f"{out}/candy-tail_check.png")


def pct(x0, y0, w, h):
    return [round(100 * x0 / W, 4), round(100 * y0 / H, 4), round(100 * w / W, 4), round(100 * h / H, 4)]


L, R = (1802, 1941), (1972, 1945)   # eyeball centres (photo coords)
ox, oy = json.load(open(f"{edir}/../master_meta.json"))["photo_offset"]
geo = {"size": [W, H], "eye_order": EYES, "tail_order": TAIL,
       "eye_patch": [epx0, epy0, epw, eph], "eye_patch_pct": pct(epx0, epy0, epw, eph),
       "tail_patch": [tpx0, tpy0, tpw, tph], "tail_patch_pct": pct(tpx0, tpy0, tpw, tph),
       "gaze_anchor": [round(((L[0] + R[0]) / 2 - ox) * sx / W, 4), round(((L[1] + R[1]) / 2 - oy) * sy / H, 4)],
       "grab_polygon_points": len(poly), "grab_polygon": poly_css}
json.dump(geo, open(f"{out}/candy-shimeji.json", "w"), indent=1)
print({k: v for k, v in geo.items() if k != "grab_polygon"})
