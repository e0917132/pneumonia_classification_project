# 소아 데이터 여부 점검용 표본 생성: (1) 무작위 200장, (2) 특징 공간 이상치 50장
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

D = next(p for p in ["data"] if os.path.exists(os.path.join(p, "train.csv")))
OUT = "analysis/age_audit"
rng = np.random.default_rng(42)
tr = pd.read_csv(os.path.join(D, "train.csv")); tr["label"] = tr.label.astype(int)
CLS = {0: "NORMAL", 1: "PNEUMONIA"}

# (1) 무작위 표본: 클래스 비율 유지 (200장)
n_total = 200
parts = []
for lab, g in tr.groupby("label"):
    k = round(n_total * len(g) / len(tr))
    parts.append(g.iloc[rng.choice(len(g), k, replace=False)])
rand = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

# (2) 이상치: ImageNet ResNet18(우리 데이터로 학습하지 않음) 특징의 kNN 평균 거리
tf = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
class DS(Dataset):
    def __len__(self): return len(tr)
    def __getitem__(self, i): return tf(Image.open(os.path.join(D, "train", tr.file_name[i])).convert("L"))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1); m.fc = nn.Identity(); m = m.to(device).eval()
with torch.no_grad():
    feats = torch.cat([m(x.to(device)).cpu() for x in DataLoader(DS(), batch_size=64)]).numpy()
f = feats - feats.mean(0); f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
sim = f @ f.T; np.fill_diagonal(sim, -1)
k = 10
knn_sim = np.sort(sim, axis=1)[:, -k:].mean(1)
tr["outlier_score"] = 1 - knn_sim                      # 클수록 다른 이미지와 멀다
out = tr.sort_values("outlier_score", ascending=False).head(50).reset_index(drop=True)

for df, name in [(rand, "random_200"), (out, "outlier_50")]:
    d = df[["file_name", "label"]].copy()
    if "outlier_score" in df: d["outlier_score"] = df.outlier_score.round(4)
    d["판단(소아/성인/불확실)"] = ""; d["근거(흉곽크기·갈비뼈·흉선·성장판·자세 등)"] = ""
    d.to_csv(f"{OUT}/{name}.csv", index=False, encoding="utf-8-sig")

def grid(df, path, title_fn, cols=10, tile=224):
    rows = int(np.ceil(len(df) / cols)); W, H = cols * tile, rows * (tile + 18)
    canvas = Image.new("L", (W, H), 255); draw = ImageDraw.Draw(canvas)
    for i, r in df.reset_index(drop=True).iterrows():
        im = Image.open(os.path.join(D, "train", r.file_name)).convert("L").resize((tile, tile))
        x, y = (i % cols) * tile, (i // cols) * (tile + 18)
        canvas.paste(im, (x, y + 18)); draw.text((x + 3, y + 3), title_fn(i, r), fill=0)
    canvas.save(path)

for g in range(4):
    sub = rand.iloc[g * 50:(g + 1) * 50]
    grid(sub, f"{OUT}/random_200_part{g+1}.png", lambda i, r, g=g: f"R{g*50+i+1:03d} {r.file_name.replace('.png','')} {CLS[r.label][:4]}")
grid(out, f"{OUT}/outlier_50.png", lambda i, r: f"O{i+1:02d} {r.file_name.replace('.png','')} {CLS[r.label][:4]} s={r.outlier_score:.3f}")
print("random:", rand.label.map(CLS).value_counts().to_dict(), "| outliers:", out.label.map(CLS).value_counts().to_dict())
print("outlier score range:", round(out.outlier_score.min(), 4), "~", round(out.outlier_score.max(), 4), "| median of all:", round(tr.outlier_score.median(), 4))
