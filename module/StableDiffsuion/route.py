import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from module.StableDiffsuion.util import run_inpaint_simple, init_inpaint_model

app = FastAPI()

# CORS 設定：允許前端 http://localhost:9000 存取
origins = ["http://localhost:9000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 使用專案根目錄的 uploads（與 main.py 共用）
PROJECT_ROOT = "./"
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 模型與設定檔路徑（可用環境變數覆蓋）
CKPT_PATH = os.getenv(
    "INPAINT_CKPT",
    os.path.join(PROJECT_ROOT, "module", "StableDiffsuion", "weight", "sd.ckpt")
)
YAML_PATH = os.getenv(
    "INPAINT_YAML",
    os.path.join(PROJECT_ROOT, "module", "StableDiffsuion", "configs", "latent-diffusion", "inpainting_example_overfit.yaml")
)
DEVICE = os.getenv("INPAINT_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
USE_EMA = bool(int(os.getenv("INPAINT_USE_EMA", "0")))
MODEL_READY = False
model = None
@app.on_event("startup")
def load_inpaint_model():
    global MODEL_READY
    # try:
    #     init_inpaint_model(ckpt_path=CKPT_PATH, yaml_profile=YAML_PATH, device=DEVICE, use_ema=USE_EMA)
    #     MODEL_READY = True
    #     print(f"[Inpaint Startup] Model loaded (device={DEVICE}, ema={USE_EMA})")
    # except Exception as e:
    #     print(f"[Inpaint Startup] Failed to load model: {e}")
    #     MODEL_READY = False
    init_inpaint_model(ckpt_path=CKPT_PATH, yaml_profile=YAML_PATH, device=DEVICE, use_ema=USE_EMA)
    MODEL_READY = True
    print(f"[Inpaint Startup] Model loaded (device={DEVICE}, ema={USE_EMA})")

@app.get("/")
async def root():
    return {"message": "StableDiffsuion inpaint service running"}

@app.get("/health")
async def health():
    status = "ready" if MODEL_READY else "not_ready"
    return {
        "status": status,
        "device": DEVICE,
        "ckpt_exists": os.path.isfile(CKPT_PATH),
        "yaml_exists": os.path.isfile(YAML_PATH)
    }

@app.post("/inpaint")
async def inpaint(original: UploadFile = File(...),
                  ref: UploadFile = File(...),
                  mask: UploadFile = File(...),
                  steps: int = 50,
                  resize: int = 512,
                  white_part: bool = False,
                  device: str = DEVICE):
    """接收 original/ref/mask 三張圖做 Inpaint，回傳修補後影像 bytes。"""
    if not MODEL_READY:
        return JSONResponse(status_code=503, content={"error": "Model not initialized"})
    try:
        original_bytes = await original.read()
        ref_bytes = await ref.read()
        mask_bytes = await mask.read()

        origin_path = os.path.join(UPLOAD_DIR, "original.jpg")
        ref_path = os.path.join(UPLOAD_DIR, "ref.jpg")
        mask_path = os.path.join(UPLOAD_DIR, "mask.jpg")

        result_bytes = run_inpaint_simple(
            origin_image_path=origin_path,
            ref_image_path=ref_path,
            mask_path=mask_path,
            steps=steps,
            resize=resize,
            white_part=white_part,
            device=device
        )
        return Response(content=result_bytes, media_type='image/jpeg')
    except Exception as e:
        raise e
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("module.StableDiffsuion.route:app", host="0.0.0.0", port=8888, reload=True)
