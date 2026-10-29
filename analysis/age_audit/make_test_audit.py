# test 데이터 연령 점검용 무작위 표본 (라벨 없음, 모델 결정에 사용하지 않음)
import os, numpy as np, pandas as pd
from PIL import Image, ImageDraw
D = next(p for p in ["data"] if os.path.exists(os.path.join(p, "test.csv")))
OUT = "analysis/age_audit"
te = pd.read_csv(os.path.join(D, "test.csv"))
samp = te.sample(100, random_state=42).reset_index(drop=True)
rows = []
for f in samp.file_name:
    w, h = Image.open(os.path.join(D, "test", f)).size; rows.append(dict(file_name=f, width=w, height=h))
d = pd.DataFrame(rows); d["판단(소아/성인/불확실)"] = ""; d["근거"] = ""
d.to_csv(f"{OUT}/test_random_100.csv", index=False, encoding="utf-8-sig")
def letterbox(img, size=224):
    w, h = img.size; s = size / max(w, h); nw, nh = round(w * s), round(h * s)
    c = Image.new("L", (size, size), 0); c.paste(img.resize((nw, nh), Image.BILINEAR), ((size - nw) // 2, (size - nh) // 2)); return c
for part in range(2):
    sub = d.iloc[part * 50:(part + 1) * 50].reset_index(drop=True); tile = 224; cols = 10
    canvas = Image.new("L", (cols * tile, 5 * (tile + 18)), 255); draw = ImageDraw.Draw(canvas)
    for i, r in sub.iterrows():
        x, y = (i % cols) * tile, (i // cols) * (tile + 18)
        canvas.paste(letterbox(Image.open(os.path.join(D, "test", r.file_name)).convert("L")), (x, y + 18))
        draw.text((x + 3, y + 3), f"T{part*50+i+1:03d} {r.file_name.replace('.png','')} {r.width}x{r.height}", fill=0)
    canvas.save(f"{OUT}/test_random_100_part{part+1}.png")
print("saved", d.shape, "| width range", d.width.min(), "-", d.width.max())
