import torch
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# COCO 클래스
DOG_CLASS_ID = 18
CAT_CLASS_ID = 17
CLASS_NAMES = {DOG_CLASS_ID: "Dog", CAT_CLASS_ID: "Cat"}

def detect_dogs_and_cats_web(image_path, save_path=None, conf_threshold=0.8):
    """
    이미지에서 강아지와 고양이를 탐지하고, 박스와 클래스명을 표시.
    conf_threshold : confidence threshold (0~1)
    """
    # 이미지 읽기
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Tensor 변환 및 배치 차원 추가
    img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)

    # 예측
    with torch.no_grad():
        outputs = model(img_tensor)

    boxes = outputs[0]['boxes'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()

    # 박스 및 클래스명 그리기
    for box, label, score in zip(boxes, labels, scores):
        if label in [DOG_CLASS_ID, CAT_CLASS_ID] and score >= conf_threshold:
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if label == DOG_CLASS_ID else (255, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{CLASS_NAMES[label]} ({score*100:.1f}%)",
                        (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 결과 저장
    if save_path:
        cv2.imwrite(save_path, img)

    return img
