"""Render tail-flick keyframes: the tail bends along its centreline, curling the tip inward (toward the body).

The extra bend grows from ~0 at the root to the most at the tip; its distribution changes per frame so the
tip lags on the way back, like a whip. The tail's floor shadow is warped with it.

usage: tail.py <master_cat.png> <master_shadow.png> <tail_spine.json> <frame_c.png> <outdir>
"""
import json
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

cat_p, shadow_p, spine_p, base_p, outdir = sys.argv[1:6]
cat = np.asarray(Image.open(cat_p).convert("RGBA")).astype(np.float32) / 255
shadow = np.asarray(Image.open(shadow_p).convert("RGBA")).astype(np.float32) / 255
base = np.asarray(Image.open(base_p).convert("RGBA")).astype(np.float32) / 255
spine = np.array(json.load(open(spine_p))["spine"], np.float32)
H, W = cat.shape[:2]

# (name, extra tip rotation in degrees, distribution exponent: low = more of the tail bends, high = tip only)
KEYFRAMES = [
    ("rest", 0.0, 2.0),
    ("up1", 10.0, 1.4),
    ("up2", 24.0, 1.7),     # peak
    ("down1", 12.0, 2.6),   # base settles first, tip still curled
    ("down2", 3.0, 3.2),
    ("over", -5.0, 2.2),    # small overshoot before settling
]
# px from the bent centreline that get carried along: the tail's half-width plus its shadow, plus however far
# that part of the tail moved (so the spot it left is redrawn as empty floor); then a fade back to untouched
REACH, REACH_PER_PX, FADE_BAND = 95.0, 1.2, 25.0
STEP = 4.0                  # spine sample spacing


def dense(points, per_step=8):
    """Resample to a fixed count (same for the original and every bent spine, so indices correspond)."""
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(points, axis=0).T))]
    s = np.linspace(0, seg[-1], (len(points) - 1) * per_step + 1)
    return np.stack([np.interp(s, seg, points[:, 0]), np.interp(s, seg, points[:, 1])], 1), s


def bend(points, amp_deg, p):
    """Add tangent rotation amp * (s/L)^p along the spine (sign: continue the tail's own curl)."""
    d = np.diff(points, axis=0)
    theta = np.arctan2(d[:, 1], d[:, 0])
    curl_sign = np.sign(theta[-1] - theta[0])
    u = np.linspace(0, 1, len(theta))
    theta2 = theta + curl_sign * np.deg2rad(amp_deg) * u ** p
    out = [points[0]]
    for t in theta2:
        out.append(out[-1] + STEP * np.array([np.cos(t), np.sin(t)]))
    return np.array(out, np.float32)


def extend(points, n=30):
    """Continue the spine straight past the tip so pixels beyond it map sensibly."""
    t = points[-1] - points[-2]
    t /= np.hypot(*t)
    return np.r_[points, points[-1] + np.arange(1, n + 1)[:, None] * STEP * t]


def normals(points):
    t = np.gradient(points, axis=0)
    t /= np.hypot(t[:, 0], t[:, 1])[:, None]
    return np.stack([-t[:, 1], t[:, 0]], 1)


src_spine = extend(spine)
src_dense, s_dense = dense(src_spine)
src_n = normals(src_dense)
yy, xx = np.mgrid[:H, :W].astype(np.float32)
pix = np.stack([xx.ravel(), yy.ravel()], 1)


def warp_maps(amp_deg, p):
    if amp_deg == 0:
        return xx, yy, np.zeros((H, W), np.float32)
    dst_spine = extend(bend(spine, amp_deg, p))
    dst_dense, _ = dense(dst_spine)
    dst_n = normals(dst_dense)
    reach = REACH + REACH_PER_PX * np.hypot(*(dst_dense - src_dense).T)
    tree = cKDTree(dst_dense)
    dist, idx = tree.query(pix, distance_upper_bound=reach.max() + FADE_BAND + 5)
    ok = np.isfinite(dist)
    idx = np.where(ok, idx, 0)
    rel = pix - dst_dense[idx]
    n_off = (rel * dst_n[idx]).sum(1)
    t_dir = np.stack([dst_n[idx][:, 1], -dst_n[idx][:, 0]], 1)
    t_off = (rel * t_dir).sum(1)            # only non-zero past the ends
    src_tan = np.stack([src_n[idx][:, 1], -src_n[idx][:, 0]], 1)
    src = src_dense[idx] + n_off[:, None] * src_n[idx] + t_off[:, None] * src_tan
    w = np.clip((reach[idx] + FADE_BAND - dist) / FADE_BAND, 0, 1) * ok
    w = w * w * (3 - 2 * w)
    mx = (pix[:, 0] + (src[:, 0] - pix[:, 0]) * w).reshape(H, W).astype(np.float32)
    my = (pix[:, 1] + (src[:, 1] - pix[:, 1]) * w).reshape(H, W).astype(np.float32)
    return mx, my, w.reshape(H, W).astype(np.float32)


def premul(img):
    out = img.copy()
    out[..., :3] *= out[..., 3:4]
    return out


cat_pm, sh_pm = premul(cat), premul(shadow)
meta = {"frames": [], "region": None}
region_any = np.zeros((H, W), bool)
for name, amp, p in KEYFRAMES:
    mx, my, w = warp_maps(amp, p)
    c = cv2.remap(cat_pm, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sh = cv2.remap(sh_pm, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    c = np.clip(c, 0, 1)
    comp = c + sh * (1 - c[..., 3:4])                     # premultiplied "over"
    a = comp[..., 3:4]
    comp_rgb = np.where(a > 1e-4, comp[..., :3] / np.maximum(a, 1e-4), 0)
    out = base.copy()
    moved = w > 0
    out[moved] = np.dstack([comp_rgb, a])[moved]
    region_any |= moved
    Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA").save(f"{outdir}/tail_{name}.png")
    tip_src = spine[-1]
    tip_dst = bend(spine, amp, p)[-1]
    meta["frames"].append({"name": name, "amp_deg": amp, "p": p, "tip_shift_px": round(float(np.hypot(*(tip_dst - tip_src))), 1)})
    print(name, "tip moves", meta["frames"][-1]["tip_shift_px"], "px", "toward", (tip_dst - tip_src).round(1))

ys, xs = np.nonzero(region_any)
meta["region"] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
json.dump(meta, open(f"{outdir}/tail_meta.json", "w"), indent=1)
print("region (master coords)", meta["region"])
