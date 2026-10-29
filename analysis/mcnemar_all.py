# 저장된 OOF 예측으로 각 변형과 비교 기준의 McNemar 검정을 다시 계산한다 (정확 이항 검정, 임계값 0.5)
import os, numpy as np, pandas as pd
from scipy.stats import binomtest
D = next(p for p in ["data"] if os.path.exists(os.path.join(p, "train.csv")))
y = pd.read_csv(os.path.join(D, "train.csv")).label.astype(int).values
oof = {"A": "analysis/oof_A.npy", "S4": "outputs/oof_S4.npy", "C1": "outputs/oof_C1.npy", "D1": "outputs/oof_D1.npy",
       "EB0": "outputs/oof_EB0.npy", "F1": "outputs/oof_F1.npy", "F2": "outputs/oof_F2.npy", "P1": "outputs/oof_P1.npy",
       "O4": "outputs/oof_O4.npy", "O11": "outputs/oof_O11.npy", "R50": "outputs/oof_R50.npy"}
pairs = [("S4", "A"), ("C1", "A"), ("D1", "C1"), ("EB0", "C1"), ("F1", "C1"), ("F2", "C1"),
         ("P1", "C1"), ("O4", "C1"), ("O11", "C1"), ("R50", "C1")]
P = {k: np.load(v) for k, v in oof.items()}
rows = []
for new, base in pairs:
    cb = (P[base] >= 0.5) == y; cn = (P[new] >= 0.5) == y
    b = int(np.sum(cb & ~cn)); c = int(np.sum(~cb & cn))
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    rows.append(dict(변형=new, 비교기준=base, 기준만_맞힘=b, 변형만_맞힘=c,
                     변형_OOF정확도=round(cn.mean(), 4), 기준_OOF정확도=round(cb.mean(), 4), p값=round(p, 4)))
res = pd.DataFrame(rows)
res.to_csv("analysis/mcnemar_results.csv", index=False, encoding="utf-8-sig")
print(res.to_string(index=False))
