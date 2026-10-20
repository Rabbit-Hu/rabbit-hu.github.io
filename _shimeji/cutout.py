"""Cut the cat out of the photo with rembg (BiRefNet) and save the raw RGBA crop.

usage: cutout.py <photo> <out.png> [model]
The crop box must match BOX in build_master.py.
"""
import sys

from PIL import Image, ImageOps
from rembg import new_session, remove

src, out = sys.argv[1:3]
model = sys.argv[3] if len(sys.argv) > 3 else "birefnet-general"
im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
crop = im.crop((800, 1580, 2300, 3110))
remove(crop, session=new_session(model)).save(out)
print("saved", out)
