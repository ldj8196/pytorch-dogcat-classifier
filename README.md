# 🐶🐱 PyTorch Dog vs Cat Object Detection

> Vibe Coding 방식으로 개발한 **PyTorch 기반 미리 학습된 Faster R-CNN 모델**을 활용하여 강아지와 고양이 이미지를 객체 탐지하는 프로젝트입니다.

---

## 📌 프로젝트 개요
- PyTorch를 이용한 CNN 모델 구현
- **Vibe Coding** 방식을 적용하여 직관적이고 유연하게 모델 설계 및 학습 진행
- pretrained=True 옵션을 통해 COCO 데이터셋으로 미리 학습된 Faster R-CNN 모델 사용.
  사람, 강아지, 고양이 등 COCO 클래스에 대해 학습된 가중치(weights)를 사용하기 때문에 별도로 학습시키지 않아도 바로 객체 탐지가 가능
  기본적으로 바로 탐지 가능 → 정확도를 높이려면 추가 학습(fine-tuning) 필요

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
- 웹에서 업로드한 이미지에서 강아지/고양이 객체 탐지
- COCO 사전 학습 Faster R-CNN 모델 사용 (pretrained=True)
- 탐지된 객체에 바운딩 박스와 클래스명 표시
- 결과 이미지 저장

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

## 🔧 향후 업그레이드 아이디어
- **DB 연동**: 학습 데이터, 이미지, 탐지 결과 등을 데이터베이스에 저장하여 관리
- **학습 내용 기록**: 학습 손실, 정확도, 모델 버전, 파라미터 등을 DB에 정리
- **웹 대시보드**: DB 기반으로 학습 기록과 추론 결과를 시각화하여 모니터링 가능
- **모델 재학습/파인튜닝 지원**: DB에 저장된 데이터로 지속적으로 모델 업데이트
