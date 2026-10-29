# 최종보고서용 그림 생성 (프로젝트과정 폴더에서 실행)
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from matplotlib import font_manager
for _f in ["AppleGothic", "Malgun Gothic", "NanumGothic"]:
    if _f in {f.name for f in font_manager.fontManager.ttflist}: plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
D = next(p for p in ["data"] if os.path.exists(os.path.join(p, "train.csv")))
OUT = "report/figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"       # 검증된 팔레트 1·2번

# 1. 실제 X-ray 샘플
tr = pd.read_csv(os.path.join(D, "train.csv")); te = pd.read_csv(os.path.join(D, "test.csv"))
nor = tr[tr.label == 0].sample(4, random_state=3).file_name.tolist()
pne = tr[tr.label == 1].sample(4, random_state=3).file_name.tolist()
tst = te.sample(4, random_state=3).file_name.tolist()
fig, axes = plt.subplots(3, 4, figsize=(11, 8.6), facecolor=SURFACE)
for r, (files, folder, name) in enumerate([(nor, "train", "train NORMAL"), (pne, "train", "train PNEUMONIA"), (tst, "test", "test 원본")]):
    for c, f in enumerate(files):
        im = Image.open(os.path.join(D, folder, f)).convert("L"); ax = axes[r, c]
        ax.imshow(im, cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(f"{name}\n{f} ({im.size[0]}×{im.size[1]})", fontsize=9, color=INK2)
fig.suptitle("데이터 샘플: train은 224×224 letterbox(검은 여백), test는 원본 해상도", color=INK, fontsize=12)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_data_samples.png", dpi=110, facecolor=SURFACE); plt.close()

# 2. 문제 해결 흐름 도식
steps = [("문제 정의", "평가 지표·규칙 확인\n가설 2개 수립"),
         ("데이터 확인", "불균형 74:26\ntrain letterbox, test 원본"),
         ("Baseline", "2층 CNN\n리더보드 0.75"),
         ("첫 개선", "사전학습 ResNet18\n검증 0.99 vs 리더보드 0.84"),
         ("평가 정비", "epoch 고정, 5-fold OOF\n리더보드 0.862"),
         ("분포 차이 대응", "누수·밝기·프레이밍 점검\n여백 크롭 0.877"),
         ("비교 실험", "모델·앙상블·층 고정\n대비 보정·Optuna"),
         ("최종 모델", "ResNet50\n리더보드 0.878"),
         ("결과 분석", "가설 검정\n실패 사례"),
         ("모델 해석", "Grad-CAM\n위험도·결과 데이터")]
fig, ax = plt.subplots(figsize=(14, 4.4), facecolor=SURFACE); ax.set_xlim(0, 5); ax.set_ylim(0, 2); ax.axis("off")
pos = [(i % 5, 1 if i < 5 else 0) for i in range(len(steps))]
for k, ((title, body), (x, yrow)) in enumerate(zip(steps, pos)):
    final = title == "최종 모델"
    box = FancyBboxPatch((x + 0.06, yrow + 0.14), 0.88, 0.72, boxstyle="round,pad=0.02,rounding_size=0.05",
                         fc="#eaf2fc" if final else "white", ec=BLUE if final else "#b8b7b1", lw=2 if final else 1)
    ax.add_patch(box)
    ax.text(x + 0.5, yrow + 0.70, f"{k+1}. {title}", ha="center", va="center", fontsize=11, color=INK, weight="bold")
    ax.text(x + 0.5, yrow + 0.40, body, ha="center", va="center", fontsize=8.6, color=INK2)
    if k < len(steps) - 1:
        nx, ny = pos[k + 1]
        if ny == yrow: ax.annotate("", (nx + 0.06, ny + 0.5), (x + 0.94, yrow + 0.5), arrowprops=dict(arrowstyle="->", color="#8a8984", lw=1.2))
        else:   # 윗줄 끝에서 아랫줄 처음으로: 아래로 → 왼쪽으로 → 아래로 꺾인 화살표
            ymid = (yrow + 0.14 + ny + 0.86) / 2
            ax.plot([x + 0.5, x + 0.5, nx + 0.5], [yrow + 0.14, ymid, ymid], color="#8a8984", lw=1.2)
            ax.annotate("", (nx + 0.5, ny + 0.86), (nx + 0.5, ymid), arrowprops=dict(arrowstyle="->", color="#8a8984", lw=1.2))
ax.set_title("문제 해결 흐름", fontsize=13, color=INK)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_flow.png", dpi=120, facecolor=SURFACE); plt.close()

# 3. 검증 정확도와 리더보드 비교 (리더보드에 제출한 실험)
rows = [("Baseline\nCNN (B5)", 0.9507, 0.7837), ("ResNet18\nfold 0", 0.9933, 0.84455), ("방법 A", 0.9885, 0.8622),
        ("C1\n여백 크롭", 0.9877, 0.8766), ("D1\nDenseNet121", 0.9887, 0.8750), ("EB0\nEffNet-B0", 0.9877, 0.8718),
        ("E1(eq)\n앙상블", 0.9893, 0.8718), ("E1\nOOF 최적", 0.9912, 0.8429), ("O4\nOptuna", 0.9883, 0.8717),
        ("O11\nOptuna", 0.9887, 0.8669), ("R50\nResNet50", 0.9875, 0.8782)]
x = np.arange(len(rows))
fig, ax = plt.subplots(figsize=(13, 5), facecolor=SURFACE); ax.set_facecolor(SURFACE)
for i, (_, v, l) in enumerate(rows): ax.plot([i, i], [l, v], color=GRID, lw=2, zorder=1)
ax.scatter(x, [r[1] for r in rows], s=70, color=BLUE, marker="o", label="검증 정확도 (OOF, ResNet18 fold 0만 fold 0 TTA)", zorder=3, edgecolor=SURFACE, linewidth=2)
ax.scatter(x, [r[2] for r in rows], s=70, color=ORANGE, marker="s", label="리더보드 정확도 (test 624장)", zorder=3, edgecolor=SURFACE, linewidth=2)
ax.annotate("최종 0.878", (x[-1], rows[-1][2]), xytext=(8, -14), textcoords="offset points", fontsize=9, color=INK)
ax.annotate("OOF 최고 0.991 → 리더보드 최저 0.843", (x[7], rows[7][2]), xytext=(-60, -24), textcoords="offset points", fontsize=9, color=INK)
ax.axhline(0.98, color="#8a8984", ls="--", lw=1); ax.text(-0.45, 0.983, "목표 0.98", ha="left", fontsize=9, color=INK2)
ax.set_xticks(x, [r[0] for r in rows], fontsize=8.5, color=INK2); ax.set_ylim(0.76, 1.005); ax.set_ylabel("정확도", color=INK2)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=INK2)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9); ax.set_title("검증 정확도와 리더보드 정확도의 차이", color=INK, fontsize=12, pad=30)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_val_vs_leaderboard.png", dpi=120, facecolor=SURFACE); plt.close()

# 4. test 이미지의 위험도별 Grad-CAM 예시 (test 라벨은 없음)
res = pd.read_csv("analysis/m5/test_results_R50.csv")
pick = []
for lv, target in [("Low", 0.0), ("Medium", 0.3), ("High", 1.0)]:
    g = res[res.risk_level == lv].copy(); g["d"] = (g.prob_pneumonia - target).abs(); pick += [(lv, r) for _, r in g.sort_values("d").head(2).iterrows()]
fig, axes = plt.subplots(2, 6, figsize=(15, 5.6), facecolor=SURFACE)
for c, (lv, r) in enumerate(pick):
    raw = Image.open(os.path.join(D, "test", r.file_name)).convert("L"); cam = Image.open(r.gradcam_path)
    axes[0, c].imshow(raw, cmap="gray"); axes[1, c].imshow(cam)
    for a in axes[:, c]: a.set_xticks([]); a.set_yticks([]); [s.set_visible(False) for s in a.spines.values()]
    axes[0, c].set_title(f"{lv} · {r.file_name}\n폐렴 확률 {r.prob_pneumonia:.2f}", fontsize=9, color=INK)
axes[0, 0].set_ylabel("test 원본", color=INK2); axes[1, 0].set_ylabel("Grad-CAM", color=INK2)
fig.suptitle("test 이미지의 위험도별 예측과 Grad-CAM (5개 fold 모델 평균, 정답 라벨 없음)", color=INK, fontsize=12)
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_test_gradcam_by_risk.png", dpi=110, facecolor=SURFACE); plt.close()
print("saved:", sorted(os.listdir(OUT)))

# 5. test 점검: 나이가 많아 보이는 9장 (비전문가 육안 판단, 정답 라벨 없음)
aud = pd.read_csv("analysis/age_audit/test_random_100.csv")
old = aud[aud["판단(소아/성인/불확실)"].str.startswith("불확실")].sort_values("R50_폐렴확률").reset_index(drop=True)
fig, axes = plt.subplots(1, len(old), figsize=(2.2 * len(old), 3.4), facecolor=SURFACE)
for ax, (_, r) in zip(axes, old.iterrows()):
    ax.imshow(Image.open(os.path.join(D, "test", r.file_name)).convert("L"), cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(f"{r.file_name.replace('.png','')}\n{r.width}×{r.height}\n폐렴 확률 {r.R50_폐렴확률:.2f} ({r.위험도})", fontsize=8.5, color=INK)
fig.suptitle("test 무작위 100장 중 나이가 더 많아 보이는 9장 (비전문가 판단, 정답 라벨 없음)", color=INK, fontsize=11)
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_test_older_looking.png", dpi=110, facecolor=SURFACE); plt.close()
print("saved fig5")
