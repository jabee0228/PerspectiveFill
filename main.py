import os
import time
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from module.LoFTR.loftr import loftrGenerate
from module.LoFTR.loftr import build_loftr_model

matcher = build_loftr_model()

app = FastAPI()

# CORS 設定：允許前端 http://localhost:9000 存取
origins = [
    "http://localhost:9000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # 開發階段也可改成 ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 在專案根目錄建立 uploads 資料夾
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/process")
async def process_images(
    original: UploadFile = File(...),
    ref: UploadFile = File(...),
    mask: UploadFile = File(...),
):
    """接收 original / ref / mask 三張圖，儲存成 jpg 檔在專案根目錄的 uploads 資料夾。"""
    # 讀取三個檔案內容
    original_bytes: bytes = await original.read()
    ref_bytes: bytes = await ref.read()
    mask_bytes: bytes = await mask.read()

    # 準備儲存路徑與檔名
    original_path = os.path.join(UPLOAD_DIR, "original.jpg")
    ref_path = os.path.join(UPLOAD_DIR, "ref.jpg")
    mask_path = os.path.join(UPLOAD_DIR, "mask.jpg")

    # 寫入檔案（以二進位方式寫入）
    with open(original_path, "wb") as f:
        f.write(original_bytes)

    with open(ref_path, "wb") as f:
        f.write(ref_bytes)

    with open(mask_path, "wb") as f:
        f.write(mask_bytes)

    # 回傳任一張或簡單訊息，這裡維持原本回傳 mask 的行為
    content_type = mask.content_type or "image/jpeg"
    loftrGenerate(matcher, original_path, ref_path, mask_path)
    return Response(content=mask_bytes, media_type=content_type)


@app.get("/")
async def root():
    return {"message": "PerspectiveFill backend is running"}


# 若直接以 python main.py 執行，啟動 uvicorn 伺服器
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
