# -*- coding: utf-8 -*-
"""Gallery: 3 hang (Goc/Blur/Sobel) x 4 cot (4 anh da the loai) -> figs/demo_gallery.png"""
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, ".."))
IMG = os.path.join(PROJ, "images")
OUT = os.path.join(IMG, "out")
FIGS = os.path.join(PROJ, "figs"); os.makedirs(FIGS, exist_ok=True)

bases = ["portrait", "wildlife", "tech", "nature"]
col_titles = ["Chân dung", "Động vật hoang dã", "Thiết bị / bàn làm việc", "Thiên nhiên"]
row_labels = ["Ảnh gốc", "Blur (r=6)", "Sobel"]

def srcs(b):
    return [os.path.join(IMG, b + ".jpg"), os.path.join(OUT, b + "_blur.png"), os.path.join(OUT, b + "_edge.png")]

C, LW, TH = 300, 145, 34
W = LW + C * len(bases)
H = TH + C * len(row_labels)
try:
    fT = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 20)
    fR = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 17)
except Exception:
    fT = fR = ImageFont.load_default()

sheet = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(sheet)
# column titles
for j, t in enumerate(col_titles):
    x = LW + j * C
    tb = d.textbbox((0, 0), t, font=fT)
    d.text((x + (C - (tb[2] - tb[0])) // 2, 7), t, fill=(20, 40, 90), font=fT)
# rows
for i in range(len(row_labels)):
    # row label (left column)
    d.text((8, TH + i * C + C // 2 - 10), row_labels[i], fill=(0, 0, 0), font=fR)
for j, b in enumerate(bases):
    imgs = srcs(b)
    for i, path in enumerate(imgs):
        im = Image.open(path).convert("RGB").resize((C, C))
        sheet.paste(im, (LW + j * C, TH + i * C))

out = os.path.join(FIGS, "demo_gallery.png")
sheet.save(out)
print("wrote", out, sheet.size)
