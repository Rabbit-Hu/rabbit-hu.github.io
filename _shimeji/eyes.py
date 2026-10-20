"""Render the gaze variants: each eyeball turns (as a sphere) behind fixed eyelids.

Each eye is split into layers built from the photo:
  iris     real iris pixels where visible; elsewhere (behind the pupil, under the lids) iris colour filled in
  pupil    one upright ellipse fitted to the photo's pupil, uniform dark, soft edge
  fixed    the lid shadow and the white highlights, which stay put while the eyeball turns
The iris and pupil rotate together; the pupil is scaled slightly so it looks the same size in every gaze
(the lids hide part of it in some). Only pixels inside the two eye openings change.

usage: eyes.py <photo> <master.png> <master_meta.json> <outdir> [taper]
Writes frame_<gaze>.png (full master canvas) for c (looking at you), n, ne, e, se, s, sw, w, nw, and
frames_meta.json with the eye region.
"""
import json
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

src, master_path, meta_path, outdir = sys.argv[1:5]
TAPER = float(sys.argv[5]) if len(sys.argv) > 5 else 0.55   # rotation strength kept right at the lid line
photo = np.asarray(ImageOps.exif_transpose(Image.open(src)).convert("RGB")).astype(np.float32) / 255
master = np.asarray(Image.open(master_path).convert("RGBA")).astype(np.float32) / 255
ox, oy = json.load(open(meta_path))["photo_offset"]

# Geometry in full-res photo coords, read off 8x zoomed crops.
EYES = {
    "L": {
        # inner edge of the dark lid rim, clockwise from the inner (left) corner
        "opening": [(1764, 1948), (1766, 1935), (1772, 1925), (1780, 1918), (1790, 1913), (1800, 1912),
                    (1810, 1913), (1820, 1916), (1828, 1923), (1834, 1932), (1837, 1943), (1837, 1953),
                    (1834, 1963), (1828, 1971), (1818, 1977), (1806, 1980), (1794, 1980), (1782, 1976),
                    (1773, 1969), (1767, 1960)],
        "centre": (1802, 1941),          # eyeball centre (pupil centre while staring at the camera)
        "pupil": ((1801.5, 1941.5), (18.5, 22.5)),   # rough centre, semi-axes of the dark oval (search zone)
        "catchlights": [((1781, 1938), 3.5, 0.55), ((1818, 1946), 5.0, 0.5), ((1836, 1951), 3.0, 0.5)],  # centre, radius, min lum
    },
    "R": {
        "opening": [(1935, 1950), (1937, 1938), (1943, 1929), (1952, 1922), (1962, 1919), (1972, 1919),
                    (1983, 1919), (1993, 1922), (2001, 1929), (2007, 1938), (2009, 1948), (2008, 1958),
                    (2003, 1968), (1995, 1976), (1984, 1980), (1971, 1982), (1958, 1982), (1947, 1978),
                    (1940, 1972), (1936, 1962)],
        "centre": (1972, 1945),
        "pupil": ((1972.0, 1945.5), (17.0, 22.0)),
        "catchlights": [((1984, 1950), 5.0, 0.5)],
    },
}
R_EYE = 46.0              # eyeball radius (px, full res)
SHIFT = (14.0, 8.0, 15.0)  # max pupil travel in px: sideways, up, down (upper lid sits close above the tall pupil)
TAPER_BAND = 8.0          # px inside the lid line over which rotation ramps from TAPER to full
PUPIL_EDGE = 4.0          # px over which the pupil fades into the iris
MAX_TILT = 4.0            # degrees the fitted pupil ellipse may lean from vertical
SHADOW_MAX = 0.2          # cap on the lid shadow taken from the photo
LID_SHADE = 0.25          # darkening of newly revealed iris right under the upper lid
LID_SHADE_BAND = 10.0     # px below the upper lid over which that shade fades out
PAD = 24                  # local crop margin around each eye

DIRS = {  # name -> screen direction (x right, y down)
    "c": None,
    "n": (0, -1), "ne": (1, -1), "e": (1, 0), "se": (1, 1),
    "s": (0, 1), "sw": (-1, 1), "w": (-1, 0), "nw": (-1, -1),
}


def soft_polygon(poly, x0, y0, w, h, ss=4):
    img = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(img).polygon([((x - x0 + 0.5) * ss, (y - y0 + 0.5) * ss) for x, y in poly], fill=255)
    return np.asarray(img.resize((w, h), Image.BOX)).astype(np.float32) / 255


def prepare(eye):
    xs, ys = zip(*eye["opening"])
    x0, y0 = min(xs) - PAD, min(ys) - PAD
    w, h = max(xs) - x0 + PAD, max(ys) - y0 + PAD
    crop = photo[y0:y0 + h, x0:x0 + w].copy()
    base = master[y0 - oy:y0 - oy + h, x0 - ox:x0 - ox + w, :3].copy()
    M = soft_polygon(eye["opening"], x0, y0, w, h)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cx, cy = eye["centre"][0] - x0, eye["centre"][1] - y0

    L3 = np.array([0.2126, 0.7152, 0.0722], np.float32)
    lum = crop @ L3
    lum_s = cv2.GaussianBlur(lum, (0, 0), 1.0)
    (pcx, pcy), (pa, pb) = eye["pupil"]
    pcx, pcy = pcx - x0, pcy - y0
    near = np.clip((1.0 - np.hypot((xx - pcx) / (pa + 4), (yy - pcy) / (pb + 4))) / 0.1, 0, 1)

    # highlights: only the bright white cores; they stay where they are
    spec = np.zeros((h, w), np.float32)
    for (sx, sy), r, lum_min in eye["catchlights"]:
        d = np.hypot(xx - (sx - x0), yy - (sy - y0))
        spec = np.maximum(spec, ((d < r * 1.5) & (lum > lum_min)).astype(np.float32))
    glint = cv2.GaussianBlur(spec, (0, 0), 0.7)[..., None]
    spec_zone = cv2.dilate(spec, np.ones((7, 7), np.uint8)) > 0

    # pupil = one ellipse fitted to the photo's pupil outline (sides and bottom; the top is cut by the lid)
    dark = np.clip((0.24 - lum_s) / 0.10, 0, 1)
    region = ((dark > 0.5) & (near > 0.5) & (M > 0.5)).astype(np.uint8)
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)[:, 0, :]
    dist_lid = cv2.distanceTransform((M > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    free = cnt[(dist_lid[cnt[:, 1], cnt[:, 0]] > 3) & ~spec_zone[cnt[:, 1], cnt[:, 0]]]
    (ecx, ecy), (ew, eh), eang = cv2.fitEllipse(free.astype(np.float32))
    eang = (eang + 90) % 180 - 90                                   # tilt from vertical, in (-90, 90]
    eang = float(np.clip(eang, -MAX_TILT, MAX_TILT))                 # cat pupils stand upright
    ea, eb, t = ew / 2, eh / 2, np.deg2rad(eang)
    u = (xx - ecx) * np.cos(t) + (yy - ecy) * np.sin(t)
    v = -(xx - ecx) * np.sin(t) + (yy - ecy) * np.cos(t)
    rn = np.hypot(u / ea, v / eb)
    # soft edge like the photo's (dark -> iris over ~4 px)
    edge = np.clip(((1.0 - rn) * min(ea, eb) + PUPIL_EDGE / 2) / PUPIL_EDGE, 0, 1)
    P = edge * edge * (3 - 2 * edge)
    P_vis = P * M
    pupil_rgb = np.median(crop[(rn < 0.8) & (dark > 0.99) & ~spec_zone], 0)
    eye["pupil_fit"] = ((ecx + x0, ecy + y0), (ea, eb), eang)

    # iris: real pixels where lit iris is visible; anywhere else (behind the pupil, under the lids, in the
    # lid shadow, around highlights) iris colour from a pull-push fill, nudged toward the typical lit iris
    inside = (M > 0.5).astype(np.uint8)
    known = ((cv2.erode(inside, np.ones((5, 5), np.uint8)) > 0) & (lum_s > 0.18)
             & (cv2.dilate((P > 0.3).astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)
             & ~spec_zone).astype(np.float32)
    fill = np.zeros_like(crop)
    done = np.zeros((h, w), bool)
    for sigma in (1.5, 3, 6, 12, 24):
        num = cv2.GaussianBlur(crop * known[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(known, (0, 0), sigma)
        take = (den > 0.05) & ~done
        fill[take] = (num / np.maximum(den, 1e-4)[..., None])[take]
        done |= take
    lit = np.median(crop[(known > 0) & (lum_s > 0.38)], 0)
    fill = cv2.GaussianBlur(fill, (0, 0), 2.0) * 0.75 + lit * 0.25
    k_soft = cv2.GaussianBlur(known, (0, 0), 1.2)[..., None]
    real = np.where(known[..., None] > 0, crop, fill)                 # don't let lid-rim pixels bleed in
    iris = real * k_soft + fill * (1 - k_soft)
    invented = 1 - k_soft[..., 0]                                     # where the iris is made up

    # lid shadow seen in the photo (dark, not pupil): fixed in place, soft and capped
    shadow_px = dark * (1 - P_vis) * M
    ratio = np.clip(lum_s / np.maximum(fill @ L3, 1e-3), 0.05, 1)
    F = 1 - np.minimum(cv2.GaussianBlur(shadow_px * (1 - ratio), (0, 0), 3.0), SHADOW_MAX)
    # plus a gentle shade right under the upper lid, for iris that was hidden there and rotates into view
    top = np.argmax(M > 0.5, axis=0).astype(np.float32)                # first row inside, per column
    lid = LID_SHADE * (1 - np.clip((yy - top[None, :]) / LID_SHADE_BAND, 0, 1)) ** 1.5 * (yy < pcy)

    dist_in = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    return dict(x0=x0, y0=y0, w=w, h=h, base=base, crop=crop, M=M[..., None], glint=glint, c=(cx, cy),
                dist_in=dist_in, iris=iris, invented=invented, P=P, pupil_rgb=pupil_rgb, pc=(pcx, pcy), F=F, lid=lid)


def eye_map(p, gaze):
    """Inverse map for a rotated eyeball: for every output pixel, where its texture came from."""
    yy, xx = np.mgrid[:p["h"], :p["w"]].astype(np.float32)
    if gaze is None:
        return xx, yy
    gx, gy = gaze
    n = np.hypot(gx, gy)
    ux, uy = gx / n, gy / n
    # travel follows a superellipse through the max shifts, so diagonals keep most of each axis
    ay = SHIFT[1] if uy < 0 else SHIFT[2]
    t = ((abs(ux) / SHIFT[0]) ** 4 + (abs(uy) / ay) ** 4) ** -0.25
    theta = np.arcsin(min(t / R_EYE, 0.95))
    axis = np.array([-uy, ux, 0.0])                       # forward (0,0,1) x gaze
    cx, cy = p["c"]
    X, Y = xx - cx, yy - cy
    Z = np.sqrt(np.maximum(R_EYE ** 2 - X ** 2 - Y ** 2, 0))
    V = np.stack([X, Y, Z], -1)
    # rotate by -theta about axis (Rodrigues): where was this surface point before the eye turned?
    c, sn = np.cos(-theta), np.sin(-theta)
    Vs = V * c + np.cross(axis, V) * sn + axis[None, None, :] * (V @ axis)[..., None] * (1 - c)
    k = TAPER + (1 - TAPER) * np.clip(p["dist_in"] / TAPER_BAND, 0, 1)
    return (xx + (Vs[..., 0] - X) * k).astype(np.float32), (yy + (Vs[..., 1] - Y) * k).astype(np.float32)


def pupil_layer(p, mapx, mapy, scale):
    pcx, pcy = p["pc"]
    sx = (pcx + (mapx - pcx) / scale).astype(np.float32)
    sy = (pcy + (mapy - pcy) / scale).astype(np.float32)
    return cv2.remap(p["P"], sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def visible_pupil(p, P):
    return float((P * p["M"][..., 0]).sum())


def render(p, gaze, target_area):
    mapx, mapy = eye_map(p, gaze)
    iris = cv2.remap(p["iris"], mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    inv = cv2.remap(p["invented"], mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # keep the apparent pupil size the same in every frame (the lids hide part of it in some gazes)
    lo, hi = 1.0, 1.08
    for _ in range(18):
        mid = (lo + hi) / 2
        if visible_pupil(p, pupil_layer(p, mapx, mapy, mid)) < target_area:
            lo = mid
        else:
            hi = mid
    scale = (lo + hi) / 2 if gaze is not None else 1.0
    P = pupil_layer(p, mapx, mapy, scale)[..., None]
    shade = p["F"] * (1 - p["lid"] * inv)        # fixed lid shadow; extra shade only on revealed iris
    eye = iris * shade[..., None]
    eye = eye * (1 - P) + p["pupil_rgb"] * P
    eye = eye * (1 - p["glint"]) + p["crop"] * p["glint"]   # highlights stay put
    return eye * p["M"] + p["base"] * (1 - p["M"]), scale


preps = {k: prepare(e) for k, e in EYES.items()}
areas = {k: visible_pupil(p, p["P"]) for k, p in preps.items()}
for dname, gaze in DIRS.items():
    out = master.copy()
    scales = []
    for k, p in preps.items():
        mx, my = p["x0"] - ox, p["y0"] - oy
        rgb, sc = render(p, gaze, areas[k])
        out[my:my + p["h"], mx:mx + p["w"], :3] = np.clip(rgb, 0, 1)
        scales.append(round(sc, 3))
    Image.fromarray((out * 255 + 0.5).astype(np.uint8), "RGBA").save(f"{outdir}/frame_{dname}.png")
    print(dname, "pupil scale L/R", scales)

x0 = min(p["x0"] for p in preps.values()) - ox
y0 = min(p["y0"] for p in preps.values()) - oy
x1 = max(p["x0"] + p["w"] for p in preps.values()) - ox
y1 = max(p["y0"] + p["h"] for p in preps.values()) - oy
json.dump({"eye_region_master": [int(x0), int(y0), int(x1), int(y1)], "dirs": list(DIRS)},
          open(f"{outdir}/frames_meta.json", "w"))
print("eye region (master coords)", x0, y0, x1, y1)
