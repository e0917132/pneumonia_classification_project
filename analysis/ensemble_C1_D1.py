# ResNet18(C1) + DenseNet121(D1) 앙상블. 가중치와 임계값은 OOF로만 정한다. test는 예측에만 쓴다.
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, roc_auc_score

D = next(p for p in ["data"] if os.path.exists(os.path.join(p, "train.csv")))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def crop_content(img, T=10):
    a = np.asarray(img.convert("L")); r, c = np.where(a > T)[0], np.where(a > T)[1]
    if len(r) == 0: return img
    return img.crop((c.min(), r.min(), c.max() + 1, r.max() + 1))
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

te = pd.read_csv(os.path.join(D, "test.csv")); y = pd.read_csv(os.path.join(D, "train.csv")).label.astype(int).values
test_imgs = [letterbox(crop_content(Image.open(os.path.join(D, "test", f)))) for f in te.file_name]
test_loader = DataLoader(DS(test_imgs), batch_size=64)

def build(name):
    if name == "resnet18":
        m = models.resnet18(weights=None); m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 2))
    else:
        m = models.densenet121(weights=None); m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier.in_features, 2))
    return m
@torch.no_grad()
def proba(m, loader):
    out = []
    for x in loader:
        x = x.to(device); p = F.softmax(m(x), 1)
        s = x.shape[-1]; c = int(s * 0.06)
        xz = F.interpolate(x[..., c:s-c, c:s-c], size=(s, s), mode="bilinear", align_corners=False)
        out.append(((p + F.softmax(m(xz), 1)) / 2)[:, 1].cpu())
    return torch.cat(out).numpy()
def test_prob(name, ckpt_fmt):
    ps = []
    for k in range(5):
        m = build(name); m.load_state_dict(torch.load(ckpt_fmt.format(k), map_location="cpu")); m = m.to(device).eval()
        ps.append(proba(m, test_loader))
    return np.mean(ps, axis=0)

t_res = test_prob("resnet18", "outputs/resnet18_C1_fold{}.pt")
t_den = test_prob("densenet121", "outputs/densenet121_D1F5_fold{}.pt")
np.save("analysis/test_prob_C1.npy", t_res); np.save("analysis/test_prob_D1.npy", t_den)

# 재현성 확인: 다시 계산한 라벨이 기존 제출 파일과 같은지
for name, tp, f in [("C1", t_res, "outputs/submission_resnet18_C1_f01234.csv"), ("D1", t_den, "outputs/submission_densenet121_D1F5_f01234.csv")]:
    sub = pd.read_csv(f); print(f"{name} 재계산 라벨 일치율: {((tp >= 0.5).astype(int) == sub.label.values).mean():.4f}")

# OOF로 가중치(DenseNet 비중)와 임계값 선택
o_res = np.load("outputs/oof_C1.npy"); o_den = np.load("outputs/oof_D1.npy")
rows = []
for w in np.round(np.arange(0, 1.01, 0.1), 1):
    p = (1 - w) * o_res + w * o_den
    for thr in np.round(np.arange(0.30, 0.71, 0.05), 2):
        rows.append(dict(w_dense=w, thr=thr, acc=accuracy_score(y, (p >= thr).astype(int)), auc=roc_auc_score(y, p)))
grid = pd.DataFrame(rows)
grid.to_csv("analysis/ensemble_grid_oof.csv", index=False)
# 동점이면 단순한 설정(가중치 0.5, 임계값 0.5)에 가까운 쪽을 고른다
grid["dist"] = (grid.w_dense - 0.5).abs() + (grid.thr - 0.5).abs()
best = grid.sort_values(["acc", "dist"], ascending=[False, True]).iloc[0]
base = grid[(grid.w_dense == 0.5) & (grid.thr == 0.5)].iloc[0]
print("OOF 단일: ResNet18", round(accuracy_score(y, o_res >= 0.5), 4), "| DenseNet121", round(accuracy_score(y, o_den >= 0.5), 4))
print("OOF 동일가중(0.5/0.5), 임계값 0.5:", round(base.acc, 4), "AUC", round(base.auc, 4))
print("OOF 최적:", dict(w_dense=best.w_dense, thr=best.thr, acc=round(best.acc, 4), auc=round(best.auc, 4)))
print("OOF 상위 5개:\n", grid.sort_values(["acc", "dist"], ascending=[False, True]).head(5)[["w_dense", "thr", "acc", "auc"]].round(4).to_string(index=False))

for tag, w, thr in [("eq", 0.5, 0.5), ("best", best.w_dense, best.thr)]:
    p = (1 - w) * t_res + w * t_den
    sub = te.copy(); sub["label"] = (p >= thr).astype(int)
    sub.to_csv(f"outputs/submission_ensemble_C1_D1_{tag}.csv", index=False)
    print(f"test 예측 분포 ({tag}, w={w}, thr={thr}):", sub.label.value_counts().sort_index().to_dict(),
          "| C1 대비 변경", int((sub.label.values != (t_res >= 0.5).astype(int)).sum()))
