import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from module.LoFTR.loftr import loftrGenerate, build_loftr_model

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

INPAINT_SERVICE = os.getenv("INPAINT_SERVICE_URL", "http://localhost:8888/inpaint")


@app.post("/process")
async def process_images(
    original: UploadFile = File(...),
    ref: UploadFile = File(...),
    mask: UploadFile = File(...),
):
    """接收 original / ref / mask 三張圖，儲存成 jpg 檔在專案根目錄的 uploads 資料夾，回傳處理結果影像。"""
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

    # 可選：先進行 LoFTR 對齊（若需要）
    loftr_bytes = loftrGenerate(matcher, original_path, ref_path, mask_path)

    # 呼叫 SD Inpaint 取得結果影像 bytes（使用剛寫入的絕對路徑避免路徑錯誤）
    # result_bytes = run_inpaint_simple(
    #     origin_image_path=original_path,
    #     ref_image_path=ref_path,
    #     mask_path=mask_path,
    #     steps=50,
    #     resize=512,
    #     white_part=False,
    #     device='cpu'  # 無 GPU 時使用 CPU；有 GPU 可改為 'cuda:0'
    # )

    # 回傳處理後影像
    return Response(content=loftr_bytes, media_type="image/jpeg")


@app.get("/")
async def root():
    return {"message": "PerspectiveFill main service running"}


# 若直接以 python main.py 執行，啟動 uvicorn 伺服器
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
