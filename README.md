# 🐶🐱 PyTorch Dog vs Cat Classifier

> Vibe Coding 방식으로 개발한 **PyTorch 기반 CNN 모델**을 활용하여 강아지와 고양이 이미지를 분류하는 프로젝트입니다.

---

## 📌 프로젝트 개요
- Kaggle **Dogs vs Cats** 데이터셋 활용
- PyTorch를 이용한 CNN 모델 구현
- **Vibe Coding** 방식을 적용하여 직관적이고 유연하게 모델 설계 및 학습 진행
- 학습 및 추론 파이프라인 구축

---

## 🛠 기술 스택
- **Framework** : PyTorch, Torchvision, FastAPI
- **Language** : Python 3.11
- **Tools** : VS Code
- **Visualization** : OpenCV
- **Web** : FastAPI, Jinja2, HTML/CSS
---

## 📦 설치된 패키지 버전 (현 프로젝트 기준)

| 패키지 | 버전 |
|--------|------|
| fastapi | 0.117.1 |
| uvicorn | 0.37.0 |
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| torchaudio | 2.6.0+cu124 |
| opencv-python | 4.12.0.88 |

## ✨ 주요 기능
- 데이터 전처리 및 증강 (Data Augmentation)
- CNN 모델 학습 및 평가
- 학습 정확도/손실 시각화
- 모델 저장(`.pth`) 및 로드 후 추론
- 웹에서 이미지 업로드 후 강아지/고양이 탐지
- 샘플 이미지 예측 결과 시각화

---

## 🖼 결과 이미지 예시
아래 이미지는 모델 추론 후 결과를 나타냅니다.

![Sample Result](resultImg.png)

---

## 🚀 실행 방법

### 1. 환경 설정
```bash
git clone https://github.com/your-username/pytorch-dogcat-classifier.git
cd pytorch-dogcat-classifier

# 가상환경 생성 및 활성화
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 필수 라이브러리 설치
pip install fastapi uvicorn[standard] opencv-python

# NVIDIA GPU가 설치되어 있다면
nvidia-smi

# 예: CUDA 12.4일 경우
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# ⚠️ 만약 GPU가 없거나 CPU만 사용하려면 +cpu 버전으로 설치하세요:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
