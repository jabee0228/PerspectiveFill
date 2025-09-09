from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import uvicorn
from module.process import process_image  # 確保這個模塊能處理三張圖片
import imgQua  # 假設這個模塊能計算 PSNR 和 SSIM

app = FastAPI()

# 啟用 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/process")
async def process(image1: UploadFile = File(...), image2: UploadFile = File(...), image3: UploadFile = File(...),
                  x1: int = Form(...), y1: int = Form(...), x2: int = Form(...), y2: int = Form(...)):
    image1_data = await image1.read()
    image1_np = np.array(Image.open(io.BytesIO(image1_data)))
    image2_data = await image2.read()
    image2_np = np.array(Image.open(io.BytesIO(image2_data)))
    image3_data = await image3.read()
    image3_np = np.array(Image.open(io.BytesIO(image3_data)))

    processed_img = process_image(image1_np, image2_np, image3_np, (x1, y1, x2, y2))
    if processed_img is None:
        return {"message": "Failed to process images", "code": 500}, 500

    output_path = "uploads/output.jpg"
    processed_img.save(output_path)

    # 儲存圖像質量指標以便後續檢索


    return {"message": "Image processed successfully"}

@app.get("/get_metrics")
async def get_metrics():
    # 假設你有一個方法來獲取最近儲存的圖像質量指標
    PSNR, SSIM = imgQua.process()  # 確保這個函數調用正確
    return JSONResponse(content={"PSNR": PSNR, "SSIM": SSIM})

@app.get("/get_image")
async def get_image():
    try:
        with open("uploads/output.jpg", "rb") as image_file:
            image = Image.open(image_file)
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="JPEG")
            img_byte_arr.seek(0)
            return StreamingResponse(img_byte_arr, media_type="image/jpeg")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Processor</title>
    </head>
    <body>
        <h1>Image Processor</h1>
        <form id="uploadForm" enctype="multipart/form-data">
            <label for="image1">Upload 底圖:</label>
            <input type="file" id="image1" name="image1" required><br><br>
            <label for="image2">Upload 修補圖:</label>
            <input type="file" id="image2" name="image2" required><br><br>
            <label for="image3">Upload 原圖:</label>
            <input type="file" id="image3" name="image3" required><br><br>
            <label for="x1">X1:</label>
            <input type="number" id="x1" name="x1" required><br><br>
            <label for="y1">Y1:</label>
            <input type="number" id="y1" name="y1" required><br><br>
            <label for="x2">X2:</label>
            <input type="number" id="x2" name="x2" required><br><br>
            <label for="y2">Y2:</label>
            <input type="number" id="y2" name="y2" required><br><br>
            <button type="button" onclick="uploadImages()">Process Images</button>
        </form>
        <h2>Processed Image:</h2>
        <img id="processedImage" src="" alt="Processed image will appear here.">
        <div id="imageMetrics"></div>
        <script>
            async function uploadImages() {
                const formData = new FormData(document.getElementById('uploadForm'));
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData,
                });

                if (response.ok) {
                    console.log('Images processed successfully');
                    getProcessedImage();
                    getImageMetrics();
                } else {
                    console.error('Failed to process images');
                }
            }

            async function getProcessedImage() {
                const response = await fetch('/get_image');
                if (response.ok) {
                    const imageBlob = await response.blob();
                    const imageUrl = URL.createObjectURL(imageBlob);
                    document.getElementById('processedImage').src = imageUrl;
                }
            }

            async function getImageMetrics() {
                const response = await fetch('/get_metrics');
                if (response.ok) {
                    const data = await response.json();
                    document.getElementById('imageMetrics').textContent = 'PSNR: ' + data.PSNR + ', SSIM: ' + data.SSIM;
                }
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
