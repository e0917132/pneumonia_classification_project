# Optuna 하이퍼파라미터 탐색: C1 설정 기준, fold 0 학습/검증만 사용 (test는 사용하지 않음)
import os, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, optuna
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

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
eval_tf = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

tr = pd.read_csv(os.path.join(D, "train.csv")); y = tr.label.astype(int).values
tr_idx, va_idx = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))[0]
imgs = [letterbox(crop_content(Image.open(os.path.join(D, "train", f)))) for f in tr.file_name]
class DS(Dataset):
    def __init__(self, idx, tf, with_y=True): self.idx, self.tf, self.with_y = idx, tf, with_y
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]; x = self.tf(imgs[j])
        return (x, int(y[j])) if self.with_y else x
va_loader = DataLoader(DS(va_idx, eval_tf), batch_size=64, shuffle=False)
counts = np.bincount(y[tr_idx], minlength=2)
class_w = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)

@torch.no_grad()
def evaluate(m):
    m.eval(); ps = []
    for x, _ in va_loader: ps.append(F.softmax(m(x.to(device)), 1)[:, 1].cpu())
    p = torch.cat(ps).numpy(); yv = y[va_idx]
    return log_loss(yv, np.clip(p, 1e-6, 1 - 1e-6)), accuracy_score(yv, p >= 0.5), roc_auc_score(yv, p)

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    wd = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    ls = trial.suggest_float("label_smoothing", 0.0, 0.1)
    epochs = trial.suggest_int("epochs", 6, 12)
    torch.manual_seed(42); np.random.seed(42)
    g = torch.Generator(); g.manual_seed(42)
    tr_loader = DataLoader(DS(tr_idx, train_tf), batch_size=32, shuffle=True, drop_last=True, generator=g)
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 2)); m = m.to(device)
    crit = nn.CrossEntropyLoss(weight=class_w, label_smoothing=ls)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    spe = len(tr_loader); total = epochs * spe; warm = spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / (total - warm))))
    t0 = time.time()
    for ep in range(epochs):
        m.train()
        for x, t in tr_loader:
            loss = crit(m(x.to(device)), t.to(device))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        ll, acc, auc = evaluate(m)
        trial.report(ll, ep)
        if trial.should_prune():
            trial.set_user_attr("acc", acc); trial.set_user_attr("auc", auc); trial.set_user_attr("pruned_at_epoch", ep + 1)
            raise optuna.TrialPruned()
    trial.set_user_attr("acc", acc); trial.set_user_attr("auc", auc); trial.set_user_attr("minutes", round((time.time() - t0) / 60, 1))
    return ll

study = optuna.create_study(study_name="C1_resnet18_fold0", storage="sqlite:///analysis/optuna_C1.db",
                            direction="minimize", load_if_exists=True,
                            sampler=optuna.samplers.TPESampler(seed=42),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
if not any(t.params == {"lr": 3e-4, "weight_decay": 1e-4, "label_smoothing": 0.05, "epochs": 12} for t in study.trials):
    study.enqueue_trial({"lr": 3e-4, "weight_decay": 1e-4, "label_smoothing": 0.05, "epochs": 12})   # 현재 C1 설정을 기준점으로 포함
remaining = 20 - len([t for t in study.trials if t.state.is_finished()])
study.optimize(objective, n_trials=max(0, remaining), timeout=150 * 60)
df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
df.to_csv("analysis/optuna_C1_trials.csv", index=False)
print("best:", study.best_trial.number, study.best_params, "logloss", round(study.best_value, 4), study.best_trial.user_attrs)
