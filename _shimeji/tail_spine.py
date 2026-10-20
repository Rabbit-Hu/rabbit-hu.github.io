"""Extract the tail centreline (root -> tip) from the cat matte, in master coords.

usage: tail_spine.py <master_cat.png> <out.json> <debug.png>
"""
import json
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

cat_path, out_json, out_dbg = sys.argv[1:4]
cat = Image.open(cat_path)
A = np.asarray(cat)[..., 3] > 127
ROOT_X = 560          # body edge where the tail leaves it

tail = A.copy()
tail[:, ROOT_X:] = False
tail[:430] = False
tail[615:, 440:] = False      # the body's left flank below the tail root
lab, n = ndi.label(tail)
tail = lab == (1 + int(np.argmax(ndi.sum(tail, lab, range(1, n + 1)))))
sk = skeletonize(tail)
ys, xs = np.nonzero(sk)
pts = {(int(x), int(y)) for x, y in zip(xs, ys)}

# walk the skeleton from its right-most point (root) along neighbours, keeping the longest path
start = max(pts, key=lambda p: p[0])
def neighbours(p):
    x, y = p
    return [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx or dy) and (x + dx, y + dy) in pts]
# BFS for farthest point (tip), then path back
prev = {start: None}
queue = [start]
for p in queue:
    for q in neighbours(p):
        if q not in prev:
            prev[q] = p
            queue.append(q)
tip = queue[-1]
path = []
p = tip
while p is not None:
    path.append(p)
    p = prev[p]
path = np.array(path[::-1], np.float32)          # root -> tip

# smooth and resample every 4 px
k = 15
pad = np.r_[np.repeat(path[:1], k, 0), path, np.repeat(path[-1:], k, 0)]
sm = np.stack([np.convolve(pad[:, i], np.ones(2 * k + 1) / (2 * k + 1), "valid") for i in range(2)], 1)
seg = np.r_[0, np.cumsum(np.hypot(*np.diff(sm, axis=0).T))]
s_new = np.arange(0, seg[-1], 4.0)
spine = np.stack([np.interp(s_new, seg, sm[:, 0]), np.interp(s_new, seg, sm[:, 1])], 1)
# near the root the skeleton hooks toward the cut line: drop that bit and continue straight into the body
straight = (spine[:, 0] > 380) & (spine[:, 0] <= 470)
spine = spine[np.argmax(spine[:, 0] <= 470):]
r_dir = spine[0] - spine[int(straight.sum())]
r_dir /= np.hypot(*r_dir)
spine = np.r_[spine[0] + np.arange(26, 0, -1)[:, None] * 4.0 * r_dir, spine]
# the skeleton stops about one half-width short of the tip: continue along the tip direction to the matte's end
t_dir = spine[-1] - spine[-6]
t_dir /= np.hypot(*t_dir)
ext = []
q = spine[-1].copy()
while True:
    q = q + 4.0 * t_dir
    if not A[int(round(q[1])), int(round(q[0]))]:
        break
    ext.append(q.copy())
spine = np.r_[spine, np.array(ext).reshape(-1, 2)]
dist = ndi.distance_transform_edt(tail)
half_w = [float(dist[int(round(y)), int(round(x))]) for x, y in spine]
json.dump({"spine": spine.round(2).tolist(), "half_width": half_w, "root_x": ROOT_X}, open(out_json, "w"))
print("spine points", len(spine), "length", round(4.0 * (len(spine) - 1), 1), "root", spine[0].round(1), "tip", spine[-1].round(1),
      "half-width min/med/max", round(min(half_w), 1), round(float(np.median(half_w)), 1), round(max(half_w), 1))

dbg = Image.new("RGBA", cat.size, (255, 255, 255, 255))
dbg.alpha_composite(cat)
d = ImageDraw.Draw(dbg)
d.line([tuple(p) for p in spine.tolist()], fill=(255, 0, 0, 255), width=3)
for i in range(0, len(spine), 10):
    x, y = spine[i]
    d.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(0, 120, 255, 255), width=2)
dbg.crop((0, 380, 720, 900)).convert("RGB").save(out_dbg)
