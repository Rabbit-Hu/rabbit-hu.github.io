"""Preview a build: eye contact sheet (3x) and an animated GIF of the tail flick (with the page's timing).

usage: preview.py <work_dir>
Writes <work_dir>/preview_eyes.png and <work_dir>/preview_tail.gif.
"""
import json
import sys

from PIL import Image, ImageDraw

work = sys.argv[1]


def on_white(im):
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im.convert("RGBA"))
    return bg.convert("RGB")


x0, y0, x1, y1 = json.load(open(f"{work}/frames/frames_meta.json"))["eye_region_master"]
tiles = []
for d in ["nw", "n", "ne", "w", "c", "e", "sw", "s", "se"]:
    t = on_white(Image.open(f"{work}/frames/frame_{d}.png").crop((x0, y0, x1, y1)))
    t = t.resize((t.width * 3, t.height * 3), Image.LANCZOS)
    ImageDraw.Draw(t).text((6, 4), d, fill=(255, 255, 0))
    tiles.append(t)
tw, th = tiles[0].size
sheet = Image.new("RGB", (tw * 3 + 8, th * 3 + 8), "white")
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % 3) * (tw + 4), (i // 3) * (th + 4)))
sheet.save(f"{work}/preview_eyes.png")

# same sequence and timing as FLICK in _includes/candy-shimeji.html (plus pauses at rest)
seq = [("rest", 600), ("up1", 60), ("up2", 150), ("down1", 90), ("down2", 80), ("over", 110), ("rest", 900)]
box = (0, 400, 720, 960)
imgs = []
for name, _ in seq:
    im = Image.open(f"{work}/tail/tail_{name}.png").crop(box)
    im = im.convert("RGBa").resize((im.width * 3 // 5, im.height * 3 // 5), Image.LANCZOS).convert("RGBA")
    imgs.append(on_white(im))
imgs[0].save(f"{work}/preview_tail.gif", save_all=True, append_images=imgs[1:], duration=[d for _, d in seq], loop=0)
print("wrote", f"{work}/preview_eyes.png", f"{work}/preview_tail.gif")
