# 확인 1: 특징 벡터를 학습 평균으로 중심화한 뒤 코사인 유사도
# 확인 2: 원본 픽셀(32x32, 표준화)의 상관계수로 거의 같은 이미지(중복) 세기
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

def pix(imgs):  # 32x32 표준화 픽셀 벡터 (상관계수 계산용)
    a = np.stack([np.asarray(im.resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel() for im in imgs])
    a = a - a.mean(1, keepdims=True); return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)

def load_feat_model(k):
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 2))
    m.load_state_dict(torch.load(f"outputs/resnet18_A_fold{k}.pt", map_location="cpu"))
    m.fc = nn.Identity(); return m.to(device).eval()

@torch.no_grad()
def embed(m, imgs):
    return torch.cat([m(x.to(device)).cpu() for x in DataLoader(DS(imgs), batch_size=64)]).numpy()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
pix_all = pix(train_imgs); pix_te = pix(test_imgs)
for k, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
    m = load_feat_model(k)
    ref = embed(m, [train_imgs[i] for i in tr_idx]); val = embed(m, [train_imgs[i] for i in va_idx]); tst = embed(m, test_imgs)
    mu = ref.mean(0)
    def cos_max(q):
        qc = q - mu; rc = ref - mu
        qc /= np.linalg.norm(qc, axis=1, keepdims=True) + 1e-8; rc /= np.linalg.norm(rc, axis=1, keepdims=True) + 1e-8
        return (qc @ rc.T).max(1)
    cv, ct = cos_max(val), cos_max(tst)
    # 픽셀 상관계수 (같은 크기 벡터이므로 내적 = 상관계수)
    pr = pix_all[tr_idx]
    pv = (pix_all[va_idx] @ pr.T).max(1); pt = (pix_te @ pr.T).max(1)
    rows.append(dict(fold=k,
        cen_cos_med_val=np.median(cv), cen_cos_med_test=np.median(ct),
        cen_cos_p95_val=np.percentile(cv, 95), cen_cos_p95_test=np.percentile(ct, 95),
        pix_corr_gt_098_val=(pv > 0.98).mean(), pix_corr_gt_098_test=(pt > 0.98).mean(),
        pix_corr_gt_095_val=(pv > 0.95).mean(), pix_corr_gt_095_test=(pt > 0.95).mean(),
        pix_med_val=np.median(pv), pix_med_test=np.median(pt)))
    print(f"fold {k} done")

res = pd.DataFrame(rows)
res.to_csv("analysis/nn_check2.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(4).to_string(index=False))
print("\n평균:\n", res.mean().round(4).to_string())
