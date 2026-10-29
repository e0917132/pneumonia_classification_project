# 최근접 이웃 확인: fold k 모델이 학습에 쓴 train 이미지 대비, 검증(val_k)과 test 이미지의 유사도 분포 비교
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold

D = next(p for p in ["data"] if os.path.exists(os.path.join(p, "train.csv")))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def letterbox(img, size=224):
    img = img.convert("L"); w, h = img.size
    if (w, h) == (size, size): return img
    s = size / max(w, h); nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = img.resize((nw, nh), Image.BILINEAR)
    c = Image.new("L", (size, size), 0); c.paste(img, ((size - nw) // 2, (size - nh) // 2)); return c

tf = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
class DS(Dataset):
    def __init__(self, imgs): self.imgs = imgs
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i): return tf(self.imgs[i])

tr = pd.read_csv(os.path.join(D, "train.csv")); te = pd.read_csv(os.path.join(D, "test.csv"))
y = tr.label.astype(int).values
train_imgs = [letterbox(Image.open(os.path.join(D, "train", f))) for f in tr.file_name]
test_imgs = [letterbox(Image.open(os.path.join(D, "test", f))) for f in te.file_name]

def load_feat_model(k):
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 2))
    m.load_state_dict(torch.load(f"outputs/resnet18_A_fold{k}.pt", map_location="cpu"))
    m.fc = nn.Identity()          # 512차원 특징 벡터 출력
    return m.to(device).eval()

@torch.no_grad()
def embed(m, imgs):
    out = []
    for x in DataLoader(DS(imgs), batch_size=64):
        out.append(F.normalize(m(x.to(device)), dim=1).cpu())
    return torch.cat(out)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for k, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
    m = load_feat_model(k)
    f_ref = embed(m, [train_imgs[i] for i in tr_idx])   # 모델 k가 학습에 쓴 이미지
    f_val = embed(m, [train_imgs[i] for i in va_idx])   # 모델 k가 보지 않은 검증 이미지
    f_te = embed(m, test_imgs)                          # 모델 k가 보지 않은 test 이미지
    nn_val = (f_val @ f_ref.T).max(1).values.numpy()    # 가장 가까운 학습 이미지와의 코사인 유사도
    nn_te = (f_te @ f_ref.T).max(1).values.numpy()
    for name, s in [("val", nn_val), ("test", nn_te)]:
        rows.append(dict(fold=k, group=name, n=len(s), median=np.median(s), p05=np.percentile(s, 5),
                         p95=np.percentile(s, 95), share_over_0p95=(s > 0.95).mean()))
    print(f"fold {k}: val median {np.median(nn_val):.3f} | test median {np.median(nn_te):.3f} | "
          f"val>0.95 {(nn_val>0.95).mean():.3f} | test>0.95 {(nn_te>0.95).mean():.3f}")

res = pd.DataFrame(rows)
res.to_csv("analysis/nn_check.csv", index=False)
print(res.groupby("group")[["median", "p05", "p95", "share_over_0p95"]].mean().round(4))
