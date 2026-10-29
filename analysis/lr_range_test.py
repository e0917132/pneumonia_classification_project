# 학습률 탐색(LR range test): C1 설정, fold 0의 학습 데이터만 사용
import os, math, numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from matplotlib import font_manager
for _f in ["AppleGothic", "NanumGothic"]:
    if _f in {f.name for f in font_manager.fontManager.ttflist}: plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False

torch.manual_seed(42); np.random.seed(42)
D = next(p for p in ["data"] if os.path.exists(os.path.join(p, "train.csv")))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
def crop_content(img, T=10):
    a = np.asarray(img.convert("L")); r, c = np.where(a > T)
    return img if len(r) == 0 else img.crop((c.min(), r.min(), c.max() + 1, r.max() + 1))
def letterbox(img, size=224):
    img = img.convert("L"); w, h = img.size
    if (w, h) == (size, size): return img
    s = size / max(w, h); nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = img.resize((nw, nh), Image.BILINEAR)
    c = Image.new("L", (size, size), 0); c.paste(img, ((size - nw) // 2, (size - nh) // 2)); return c
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.06, 0.06), scale=(0.9, 1.1), fill=0),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
tr = pd.read_csv(os.path.join(D, "train.csv")); y = tr.label.astype(int).values
tr_idx, _ = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))[0]
imgs = [letterbox(crop_content(Image.open(os.path.join(D, "train", tr.file_name[i])))) for i in tr_idx]
labels = y[tr_idx]
class DS(Dataset):
    def __len__(self): return len(imgs)
    def __getitem__(self, i): return train_tf(imgs[i]), int(labels[i])
loader = DataLoader(DS(), batch_size=32, shuffle=True, drop_last=True)

m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 2)); m = m.to(device)
counts = np.bincount(labels, minlength=2)
crit = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device), label_smoothing=0.05)
lr_min, lr_max, n_steps = 1e-7, 1.0, 2 * len(loader)
opt = torch.optim.AdamW(m.parameters(), lr=lr_min, weight_decay=1e-4)
mult = (lr_max / lr_min) ** (1 / n_steps)
lrs, losses, avg, best, step = [], [], 0.0, float("inf"), 0
m.train()
done = False
while not done:
    for x, t in loader:
        lr = lr_min * mult ** step
        for g in opt.param_groups: g["lr"] = lr
        loss = crit(m(x.to(device)), t.to(device))
        opt.zero_grad(); loss.backward(); opt.step()
        avg = 0.9 * avg + 0.1 * loss.item(); sm = avg / (1 - 0.9 ** (step + 1))   # 지수 이동 평균
        lrs.append(lr); losses.append(sm); best = min(best, sm); step += 1
        if sm > 4 * best or step >= n_steps: done = True; break

lrs, losses = np.array(lrs), np.array(losses)
i_min = int(np.argmin(losses))
grad = np.gradient(losses, np.log10(lrs))
lo = max(5, int(0.05 * len(lrs)))
i_steep = lo + int(np.argmin(grad[lo:i_min])) if i_min > lo else i_min
pd.DataFrame({"lr": lrs, "smoothed_loss": losses}).to_csv("analysis/lr_range_test.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(lrs, losses); ax.set_xscale("log")
from matplotlib.ticker import FuncFormatter, LogLocator
ax.xaxis.set_major_locator(LogLocator(base=10)); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"1e{int(round(np.log10(v)))}")); ax.set_xlabel("학습률 (log)"); ax.set_ylabel("손실 (이동 평균)")
ax.axvline(lrs[i_steep], ls="--", color="green", label=f"가장 가파른 감소 {lrs[i_steep]:.1e}")
ax.axvline(lrs[i_min], ls="--", color="red", label=f"최소 손실 {lrs[i_min]:.1e}")
ax.axvline(3e-4, ls=":", color="gray", label="현재 C1 학습률 3e-4")
ax.set_title("학습률 탐색 (C1 설정, fold 0 학습 데이터)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("analysis/lr_range_test.png", dpi=110)
print(f"steps {step} | 가장 가파른 감소 lr={lrs[i_steep]:.2e} | 최소 손실 lr={lrs[i_min]:.2e} (loss {losses[i_min]:.3f}) | 권장 상한(최소/10) {lrs[i_min]/10:.2e}")
