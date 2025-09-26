import os
import uuid
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from model_utils import detect_dogs_and_cats_web

app = FastAPI()

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "static", "result")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# 필요한 폴더가 없으면 생성
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict_image(request: Request, file: UploadFile = File(None)):
    error_msg = None

    # 첨부 파일 체크
    if file is None:
        error_msg = "파일이 첨부되지 않았습니다. 이미지를 업로드해주세요."
        return templates.TemplateResponse("index.html", {"request": request, "error_msg": error_msg})

    # 파일 확장자 체크
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        error_msg = "지원하지 않는 파일 형식입니다. JPG, PNG 등 이미지 파일을 업로드해주세요."
        return templates.TemplateResponse("index.html", {"request": request, "error_msg": error_msg})

    # 안전한 파일명 생성
    safe_filename = f"{uuid.uuid4().hex}{ext}"
    upload_path = os.path.join(UPLOAD_DIR, safe_filename)
    result_path = os.path.join(RESULT_DIR, safe_filename)

    # 파일 저장
    try:
        with open(upload_path, "wb") as f:
            f.write(await file.read())
    except Exception:
        error_msg = "파일 저장 중 오류가 발생했습니다. 다시 시도해주세요."
        return templates.TemplateResponse("index.html", {"request": request, "error_msg": error_msg})

    # OpenCV로 이미지 읽기
    import cv2
    img = cv2.imread(upload_path)
    if img is None:
        error_msg = "이미지를 읽을 수 없습니다. 다른 이미지를 업로드해주세요."
        return templates.TemplateResponse("index.html", {"request": request, "error_msg": error_msg})

    # 정상 이미지일 때 탐지 수행
    try:
        detect_dogs_and_cats_web(upload_path, save_path=result_path)
    except Exception:
        error_msg = "이미지 탐지 중 오류가 발생했습니다."
        return templates.TemplateResponse("index.html", {"request": request, "error_msg": error_msg})

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result_img": f"/static/result/{safe_filename}"
    })
