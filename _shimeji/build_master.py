"""Build the master RGBA: the cat (cut out by cutout.py) plus the real floor shadow measured from the photo.

usage: build_master.py <photo> <cutout.png> <outdir> [shadow_gain]
Writes
  master.png         cat composited over its shadow (what the page shows)
  master_cat.png     cat only (same canvas)
  master_shadow.png  shadow only, also under the cat (so the tail's shadow can move with it)
  master_meta.json   canvas offset in the full-res photo
"""
import json
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

src, cut, outdir = sys.argv[1:4]
BOX = (800, 1580, 2300, 3110)  # crop used for the rembg run
HULL_Y0, HULL_X0, HULL_X1 = 700, 600, 1140  # (crop coords) lower-body region for the left shadow hull
GAP_Y0 = 1050                                # (crop coords) row where the forelegs separate

photo = np.asarray(ImageOps.exif_transpose(Image.open(src)).convert("RGB").crop(BOX)).astype(np.float32) / 255
A = np.asarray(Image.open(cut))[..., 3].astype(np.float32) / 255
rgb = photo.copy()

def nblur(v, m, sigma):
    """Gaussian blur of v using only pixels where m is set (normalized convolution)."""
    if v.ndim == 3:
        return np.stack([nblur(v[..., c], m, sigma) for c in range(v.shape[2])], -1)
    return ndi.gaussian_filter(v * m, sigma) / np.maximum(ndi.gaussian_filter(m, sigma), 1e-4)

# floor samples skip a band around the cat: faint fur tips there have alpha ~0 but are bright
near_cat = ndi.distance_transform_edt(A < 0.02) < 4
floor = ((A < 0.02) & ~near_cat).astype(np.float32)

# --- edge colour un-mixing: photo = A*fur + (1-A)*carpet; solve for fur using the local carpet colour
carpet = nblur(rgb, floor, 8)
a3 = A[..., None]
unmixed = np.clip((rgb - (1 - a3) * carpet) / np.maximum(a3, 1e-3), 0, 1)
core = (A > 0.97).astype(np.float32)
fur_near = nblur(rgb, core, 5)  # fallback where alpha is too small for a stable solve
t = np.clip((A - 0.15) / 0.35, 0, 1)[..., None]
edge_rgb = unmixed * t + fur_near * (1 - t)
cat_rgb = np.where(a3 > 0.97, rgb, edge_rgb)

# --- shadow: how much darker the carpet is than lit carpet
L = photo @ np.array([0.2126, 0.7152, 0.0722], np.float32)
# smooth carpet luminance (texture removed), ignoring cat pixels
Ls = nblur(L, floor, 9)
# lit reference: large-scale average over clearly lit, far-from-cat carpet
far = (ndi.distance_transform_edt(A < 0.5) > 140) & (Ls > 0.52)
L0 = ndi.gaussian_filter(L * far, 220) / np.maximum(ndi.gaussian_filter(far.astype(np.float32), 220), 1e-4)
shade = np.clip(1 - Ls / L0, 0, 1)

# The shadow *shape* comes from the photo, but only where the photo can be trusted:
# the floor on the image right is also darkened by an obstacle outside the frame (the tunnel),
# so there we only add a faint synthetic contact shadow.
#   measured: notch between the cat's right foreleg (image left) and her haunch, which sits under
#             the body (convex hull of the lower body left of the other foreleg); the gap between
#             the forelegs under the chest (fading toward the paws); a contact rim on the image left
#             (haunch, tail).
#   synthetic: thin contact shadow where legs and paws meet the floor, both sides.
body = A > 0.5
rim = ndi.distance_transform_edt(~body)
H, W = A.shape
Y, X = np.mgrid[:H, :W].astype(np.float32)
def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)

disk_r = 70
yy, xx = np.mgrid[:disk_r * 2 + 1, :disk_r * 2 + 1]
disk = (yy - disk_r) ** 2 + (xx - disk_r) ** 2 <= disk_r ** 2
gap = ndi.binary_closing(np.pad(body, disk_r), structure=disk)[disk_r:-disk_r, disk_r:-disk_r] & ~body
lower = body.copy()
lower[:HULL_Y0] = False   # above this row the tail is still attached to the body
lower[:, :HULL_X0] = False
lower[:, HULL_X1:] = False
ys, xs = np.nonzero(lower)
pts = np.c_[xs, ys]
hull = pts[ConvexHull(pts).vertices]
hmask = Image.new("L", (W, H), 0)
ImageDraw.Draw(hmask).polygon([tuple(p) for p in hull.tolist()], fill=255)
notch = (np.asarray(hmask) > 0) & ~body

def feathered(mask, r=45):
    e = smoothstep(0, r, ndi.distance_transform_edt(mask | body))
    w = mask * np.maximum(e, np.clip(1 - rim / 40, 0, 1))
    # let the mask continue under the adjacent fur so the blur doesn't leave a light seam at the body edge
    under = body & ndi.binary_dilation(mask, iterations=25)
    return ndi.gaussian_filter(np.maximum(w, under.astype(np.float32)), 10)

left = 1 - smoothstep(HULL_X1 - 60, HULL_X1 + 20, X)          # trust the photo only on the image left
gap_fade = 1 - 0.55 * smoothstep(GAP_Y0, GAP_Y0 + 330, Y)       # under chest -> between paws
contact_left = ndi.gaussian_filter(np.clip(1 - rim / 30, 0, 1) ** 1.5, 4) * left
measured_region = np.maximum.reduce([feathered(notch), feathered(gap) * gap_fade, contact_left])

gain = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
shade_s = ndi.gaussian_filter(shade, 10)  # drop carpet mottling
synthetic = 0.3 * ndi.gaussian_filter(np.clip(1 - rim / 22, 0, 1) ** 1.5, 3) * smoothstep(GAP_Y0 - 150, GAP_Y0, Y)
S = np.clip(np.maximum(shade_s * gain * measured_region, synthetic), 0, 0.75)

# --- composite cat over shadow (shadow colour: warm near-black)
K = np.array([28, 24, 20], np.float32) / 255
a_out = A + S * (1 - A)
c_out = (cat_rgb * A[..., None] + K * (S * (1 - A))[..., None]) / np.maximum(a_out, 1e-5)[..., None]

ys, xs = np.nonzero(a_out > 0.01)
pad = 4
x0, x1, y0, y1 = xs.min() - pad, xs.max() + pad + 1, ys.min() - pad, ys.max() + pad + 1
def save(c, a, path):
    arr = np.dstack([c, a])[y0:y1, x0:x1]
    Image.fromarray(np.clip(arr * 255 + 0.5, 0, 255).astype(np.uint8), "RGBA").save(path)
save(c_out, a_out, f"{outdir}/master.png")
save(np.broadcast_to(K, c_out.shape), S, f"{outdir}/master_shadow.png")  # shadow layer alone (also under the cat)
save(cat_rgb, A, f"{outdir}/master_cat.png")
meta = {"photo_offset": [int(BOX[0] + x0), int(BOX[1] + y0)], "size": [int(x1 - x0), int(y1 - y0)]}
json.dump(meta, open(f"{outdir}/master_meta.json", "w"))
print(meta)
