# 흉부 X-ray 이미지 분류 프로젝트

소아 흉부 X-ray를 정상(NORMAL)과 폐렴(PNEUMONIA)으로 분류하는 딥러닝 모델을 개발하고, 검증 점수와 리더보드 점수의 격차를 분석한 프로젝트이다.

> [!IMPORTANT]
> **최종 모델 R50** (ImageNet 사전학습 ResNet50, 여백 크롭 + letterbox, 5-fold) · 리더보드 **0.8782** (548/624) · Baseline 0.75 대비 **+80장** · 목표 0.98 미달

## 결과 요약

| 단계 | 변경점 | OOF 정확도 | 리더보드 (맞힌 장수) |
|---|---|---|---|
| Baseline | 2층 CNN, 128×128 Resize, 2 epoch | - | 0.75 (468) |
| ResNet18 fold 0 | 사전학습, letterbox, 증강, 클래스 가중치 | 0.9933 (fold 0, TTA) | 0.8446 (527) |
| 방법 A | 12 epoch 고정, 5-fold 평균 | 0.9885 | 0.8622 (538) |
| C1 | 검은 여백 크롭 | 0.9877 | 0.8766 (547) |
| D1 | DenseNet121 | 0.9887 | 0.8750 (546) |
| E1 | C1 + D1 앙상블 (OOF 최적) | 0.9912 | 0.8429 (526) |
| **R50** | **ResNet50 (최종)** | 0.9875 | **0.8782 (548)** |

전체 실험 14개와 McNemar 검정 결과는 [Mission 0\~5 정리](docs/missions.md#mission-3-실험-수행-및-검증)에 있다.

![검증 정확도와 리더보드 정확도](report/figures/fig3_val_vs_leaderboard.png)

## 핵심 발견

1. **검증 점수가 실제 성능을 보장하지 않았다.** 검증 정확도는 0.98\~0.99였지만 리더보드는 0.84\~0.88에 머물렀다. OOF가 가장 높은 앙상블이 리더보드는 가장 낮았다.
2. **모델이 폐 바깥의 단서에 의존하는 경향을 보였다.** Grad-CAM에서 폐렴 판단은 어깨, 촬영 표시, 배, 이미지 가장자리를 보는 경우가 많았다. 이런 지름길 단서는 격차에 대한 가능한 설명 중 하나이며, 원인으로 확정하지는 않았다.
3. **개선된 전체 파이프라인은 Baseline보다 크게 나았다**(McNemar p≈10⁻³⁵). 그러나 이후의 모델 교체, 증강, 앙상블, 하이퍼파라미터 조정은 검증에서 유의한 차이를 만들지 못했다.

![확신 있게 맞힌 이미지의 Grad-CAM](analysis/m5/gradcam_correct.png)

## 접근 방법

| 구분 | 내용 |
|---|---|
| 전처리 | 검은 여백 크롭(임계값 T=10, train에서만 결정) → letterbox 224×224, 좌우 반전 없는 약한 증강 |
| 모델 | ImageNet 사전학습 ResNet18 / DenseNet121 / EfficientNet-B0 / ResNet50 비교 |
| 학습 | 클래스 가중치, label smoothing 0.05, AdamW + warmup·Cosine LR, 12 epoch 고정 |
| 검증 | Stratified 5-fold OOF, 실험 간 McNemar 검정, 임계값 0.5 고정 |
| 분석 | 실패 사례 분석, Grad-CAM, 위험도 3단계(Low / Medium / High, 연구용 제안) |

## 실행 방법

```bash
pip install -r requirements.txt
```

1. 데이콘에서 받은 데이터를 저장소 루트의 `data/`에 둔다 (`train.csv`, `test.csv`, `sample_submission.csv`, `train/`, `test/`).
2. `notebooks/final/R50_ResNet50_5fold.ipynb`를 실행한다. 체크포인트, OOF 예측, 제출 파일은 `outputs/`에 저장된다.
3. `notebooks/final/Mission4_결과분석.ipynb`, `Mission5_GradCAM_결과데이터.ipynb` 순서로 실행한다.
4. `analysis/`의 스크립트는 저장소 루트에서 실행한다 (예: `python analysis/mcnemar_all.py`).

Apple Silicon(MPS)에서 실행했다. CUDA와 CPU에서도 동작하지만 결과 수치는 조금 다를 수 있다.

## 문서

| 문서 | 내용 |
|---|---|
| [최종 보고서](report/최종보고서.md) ([PDF](report/최종보고서.pdf)) | 문제 정의부터 실험, 분석, 서비스 설계, 한계까지 |
| [Mission 0\~5 정리](docs/missions.md) | 미션별 상세 정리와 Reflection 답변 |
| [데일리 로그](docs/daily-logs/) | 단계별 진행 기록 (M1\~M5) |
| [실험 기록](analysis/실험기록.csv) | 실험별 OOF와 리더보드 |

## 데이터

- Kermany, D., Zhang, K., Goldbaum, M. (2018). *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*, Mendeley Data, V2. https://doi.org/10.17632/rscbjbr9sj.2
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- 학습에는 데이콘 해커톤에서 제공한 데이터(train 5,216장, test 624장)만 사용했으며, 데이터는 저장소에 포함하지 않는다.
- 그림과 노트북 결과에는 위 데이터의 흉부 X-ray 이미지가 크롭·크기 조정·Grad-CAM 겹치기 등으로 변형되어 포함되어 있다.
