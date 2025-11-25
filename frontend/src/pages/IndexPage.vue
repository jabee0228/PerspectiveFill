<template>
  <q-page>


    <!-- 主要內容區域 -->
    <div class="q-pa-lg">
      <div class="row q-gutter-lg flex-nowrap items-stretch">
        <!-- 左側上傳區域 -->
        <div class="col-12 col-md-2">
          <q-card class="upload-card" elevated>
            <q-card-section class="text-center">
              <q-icon name="cloud_upload" size="4rem" color="primary" />
              <div class="text-h6 q-mt-md">上傳圖片</div>
              <div class="text-caption text-grey-6 q-mb-md">
                支援 JPG, PNG, WEBP 格式
              </div>

              <q-btn
                color="primary"
                icon="add_photo_alternate"
                label="選擇圖片"
                size="lg"
                class="q-mt-md"
                @click="selectImage"
              />

              <!-- 拖拽上傳區域 -->
              <div
                class="drag-area q-mt-lg q-pa-lg"
                @dragover.prevent
                @drop.prevent="handleDrop"
              >
                <q-icon name="file_upload" size="2rem" color="grey-5" />
                <div class="text-body2 text-grey-6 q-mt-sm">
                  或拖拽圖片到此處
                </div>
              </div>
            </q-card-section>
          </q-card>

          <!-- 參考圖上傳區域 -->
          <q-card class="reference-card q-mt-md" elevated>
            <q-card-section class="text-center">
              <div class="text-h6 q-mb-sm">參考圖上傳</div>
              <div class="text-caption text-grey-6 q-mb-md">
                僅使用自訂義參考圖
              </div>

              <q-btn
                color="primary"
                icon="image_search"
                label="選擇參考圖"
                size="md"
                @click="selectReferenceImage"
              />

              <div
                class="drag-area q-mt-md q-pa-md"
                @dragover.prevent
                @drop.prevent="handleRefDrop"
              >
                <q-icon name="file_upload" size="1.8rem" color="grey-5" />
                <div class="text-body2 text-grey-6 q-mt-sm">
                  或拖拽參考圖到此處
                </div>
              </div>

              <div v-if="referenceImage" class="q-mt-md">
                <q-img :src="referenceImage" ratio="1" class="ref-preview" :img-style="{ objectFit: 'cover' }" />
                <q-btn flat color="negative" icon="clear" label="清除參考圖" class="q-mt-sm" @click="referenceImage = null" />
              </div>
            </q-card-section>
          </q-card>
        </div>



        <!-- 右側預覽和操作區域 -->
        <div class="col-12 col-md-6">
          <q-card elevated>
            <q-card-section>
              <div class="text-h6 q-mb-md">圖片預覽</div>

              <!-- 畫筆大小控制 -->
              <div v-if="selectedImage" class="q-mb-md">
                <q-slider
                  v-model="brushSize"
                  :min="5"
                  :max="50"
                  :step="1"
                  label
                  :label-value="`畫筆大小: ${brushSize}px`"
                  color="primary"
                  class="q-mb-sm"
                />
              </div>

              <!-- 圖片預覽區域 -->
              <div class="preview-area" style="position: relative;">
                <div v-if="!selectedImage" class="no-image">
                  <q-icon name="image" size="5rem" color="grey-4" />
                  <div class="text-body1 text-grey-6 q-mt-md">
                    請選擇要修補的圖片
                  </div>
                </div>

                <!-- 主圖顯示 -->
                <div v-else class="canvas-container" style="position: relative;">
                  <!-- 顯示影像與半透明紅色筆畫的主畫布 -->
                  <canvas
                    ref="canvas"
                    class="drawing-canvas"
                    @mousedown="startDrawing"
                    @mousemove="draw"
                    @mouseup="stopDrawing"
                    @mouseleave="stopDrawing"
                  />

                  <!-- 修補中灰色遮罩 -->
                  <div
                    v-if="isRepairing"
                    style="position:absolute;top:0;left:0;width:100%;height:100%;background:rgba(80,80,80,0.5);z-index:2;display:flex;align-items:center;justify-content:center;"
                  >
                    <div style="width:80%;position:absolute;bottom:24px;left:10%;">
                      <q-linear-progress
                        :value="repairProgress/100"
                        color="primary"
                        track-color="grey-4"
                        rounded
                        size="20px"
                      />
                    </div>
                  </div>

                  <!-- 修補成功提示 -->
                  <transition name="fade-success">
                    <div v-if="repairSuccess" class="repair-success-mask">
                      <div class="bg-white text-primary text-h5 q-pa-xl q-mb-xl repair-success-tip">
                        修補成功！
                      </div>
                    </div>
                  </transition>
                </div>
              </div>
            </q-card-section>

            <!-- 操作按鈕 -->
            <q-card-actions v-if="selectedImage" align="right">
              <q-btn
                flat
                color="secondary"
                icon="undo"
                label="重置遮罩"
                @click="resetCanvas"
              />
              <q-btn
                flat
                color="negative"
                icon="clear"
                label="清除"
                @click="clearImage"
              />
              <q-btn
                color="primary"
                icon="auto_fix_high"
                label="開始修補"
                @click="startRepair"
              />
            </q-card-actions>
          </q-card>
        </div>
      </div>
    </div>
  </q-page>
</template>

<script setup lang="ts">
import { ref, nextTick, watch } from 'vue';
import { useRouter } from 'vue-router';

const activeTab = ref('repair');
const selectedImage = ref<string | null>(null);
const referenceImage = ref<string | null>(null);
const canvas = ref<HTMLCanvasElement | null>(null);
const maskCanvas = ref<HTMLCanvasElement | null>(null);

const isDrawing = ref(false);
const brushSize = ref(20);
const isRepairing = ref(false);
const repairProgress = ref(0);
const repairSuccess = ref(false);

let ctx: CanvasRenderingContext2D | null = null; // 顯示畫布 context
let maskCtx: CanvasRenderingContext2D | null = null; // mask 畫布 context

const router = useRouter();

const selectImage = () => {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/*';
  input.onchange = (event) => {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        selectedImage.value = e.target?.result as string;
        void nextTick(() => {
          setupCanvas();
        });
      };
      reader.readAsDataURL(file);
    }
  };
  input.click();
};

const handleDrop = (event: DragEvent) => {
  const files = event.dataTransfer?.files;
  if (files && files.length > 0) {
    const file = files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        selectedImage.value = e.target?.result as string;
        void nextTick(() => {
          setupCanvas();
        });
      };
      reader.readAsDataURL(file);
    }
  }
};

// 新增：選擇/拖拽 參考圖
const selectReferenceImage = () => {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/*';
  input.onchange = (event) => {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        referenceImage.value = e.target?.result as string;
      };
      reader.readAsDataURL(file);
    }
  };
  input.click();
};

const handleRefDrop = (event: DragEvent) => {
  const files = event.dataTransfer?.files;
  if (files && files.length > 0) {
    const file = files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        referenceImage.value = e.target?.result as string;
      };
      reader.readAsDataURL(file);
    }
  }
};

// 初始化主畫布與 mask 畫布（同解析度）
const setupCanvas = () => {
  if (!canvas.value || !selectedImage.value) return;

  const img = new Image();
  img.onload = () => {
    const canvasEl = canvas.value!;
    ctx = canvasEl.getContext('2d');
    if (!ctx) return;

    const containerWidth = canvasEl.parentElement!.clientWidth;
    const scale = Math.min(containerWidth / img.width, 400 / img.height);

    const targetWidth = img.width * scale;
    const targetHeight = img.height * scale;

    canvasEl.width = targetWidth;
    canvasEl.height = targetHeight;

    // 建立 / 重設 mask 畫布（同解析度）
    if (!maskCanvas.value) {
      maskCanvas.value = document.createElement('canvas');
    }
    const mCanvas = maskCanvas.value;
    mCanvas.width = targetWidth;
    mCanvas.height = targetHeight;
    maskCtx = mCanvas.getContext('2d');
    if (maskCtx) {
      // 初始為全黑（無筆畫）
      maskCtx.fillStyle = 'black';
      maskCtx.fillRect(0, 0, targetWidth, targetHeight);
    }

    // 畫原圖到底層
    ctx.clearRect(0, 0, targetWidth, targetHeight);
    ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
  };
  img.src = selectedImage.value;
};

// 回傳目前 mask 圖（同解析度）的 DataURL，可給後端使用
const getMaskDataUrl = () => {
  if (!maskCanvas.value) return null;
  return maskCanvas.value.toDataURL('image/png');
};

// 取得滑鼠在畫布上的位置
const getCanvasPos = (event: MouseEvent) => {
  const rect = canvas.value!.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  return { x, y };
};

const startDrawing = (event: MouseEvent) => {
  if (!ctx || !maskCtx || !canvas.value) return;

  isDrawing.value = true;

  const { x, y } = getCanvasPos(event);

  // 顯示畫布：半透明紅色 50%
  ctx.globalCompositeOperation = 'source-over';
  ctx.strokeStyle = 'rgba(255,0,0,0.5)';
  ctx.lineWidth = brushSize.value;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(x, y);

  // mask 畫布：白色（不透明）
  maskCtx.globalCompositeOperation = 'source-over';
  maskCtx.strokeStyle = 'white';
  maskCtx.lineWidth = brushSize.value;
  maskCtx.lineCap = 'round';
  maskCtx.lineJoin = 'round';
  maskCtx.beginPath();
  maskCtx.moveTo(x, y);
};

const draw = (event: MouseEvent) => {
  if (!isDrawing.value || !ctx || !maskCtx || !canvas.value) return;

  const { x, y } = getCanvasPos(event);

  ctx.lineTo(x, y);
  ctx.stroke();

  maskCtx.lineTo(x, y);
  maskCtx.stroke();
};

const stopDrawing = () => {
  if (!isDrawing.value) return;

  isDrawing.value = false;

  if (ctx) {
    ctx.beginPath();
  }
  if (maskCtx) {
    maskCtx.beginPath();
  }
};

const resetCanvas = () => {
  if (selectedImage.value) {
    setupCanvas();
  }
};

const clearImage = () => {
  selectedImage.value = null;
  ctx = null;
  maskCtx = null;
  maskCanvas.value = null;
};

// 將 dataURL 轉成 File 物件，方便用 FormData 上傳
// 修正 undefined 可能性：加入嚴格檢查並確保 mime 一定是 string
const dataURLToFile = (dataUrl: string, filename: string): File => {
  const parts = dataUrl.split(',');
  const meta = parts[0];
  const base64 = parts[1];
  if (!meta || !base64) {
    throw new Error('Invalid data URL format');
  }
  const mimeMatch = meta.match(/:(.*?);/);
  const mime: string = mimeMatch?.[1] ?? 'image/png';
  const bstr = atob(base64);
  const len = bstr.length;
  const u8arr = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    u8arr[i] = bstr.charCodeAt(i);
  }
  return new File([u8arr], filename, { type: mime });
};

// 將目前選擇的圖片 (dataURL or URL) 取回成 Blob，用於組 FormData
const urlOrDataUrlToFile = async (src: string, filename: string) => {
  // 若已經是 dataURL，直接用上面的工具轉
  if (src.startsWith('data:')) {
    return dataURLToFile(src, filename);
  }
  // 若是一般 URL（例如後端回傳或 public 資源），先 fetch 再轉成 File
  const res = await fetch(src);
  const blob = await res.blob();
  return new File([blob], filename, { type: blob.type || 'image/jpeg' });
};

// 與後端串接的真正修補流程
const startRepair = async () => {
  if (!selectedImage.value || !referenceImage.value) {
    console.warn('缺少原圖或參考圖，無法送後端');
    return;
  }

  const maskDataUrl = getMaskDataUrl();
  if (!maskDataUrl) {
    console.warn('mask 尚未建立，無法送後端');
    return;
  }

  isRepairing.value = true;
  repairSuccess.value = false;
  repairProgress.value = 0;

  try {
    // 1. 準備 FormData
    const formData = new FormData();

    const originalFile = await urlOrDataUrlToFile(selectedImage.value, 'original.png');
    const refFile = await urlOrDataUrlToFile(referenceImage.value, 'ref.png');
    const maskFile = dataURLToFile(maskDataUrl, 'mask.png');

    formData.append('original', originalFile);
    formData.append('ref', refFile);
    formData.append('mask', maskFile);

    // 2. 呼叫後端 API
    const response = await fetch('http://localhost:8000/process', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`後端回傳錯誤狀態碼: ${response.status}`);
    }

    // 3. 取得後端回傳的圖片（假設直接回傳圖片檔）
    const blob = await response.blob();
    const resultUrl = URL.createObjectURL(blob);

    // 將預覽圖換成後端結果
    selectedImage.value = resultUrl;

    // 更新畫布顯示
    await nextTick();
    setupCanvas();

    // 顯示成功訊息（可依喜好調整）
    repairProgress.value = 100;
    repairSuccess.value = true;
    setTimeout(() => {
      repairSuccess.value = false;
    }, 2000);
  } catch (err) {
    console.error('修補過程發生錯誤: ', err);
  } finally {
    isRepairing.value = false;
  }
};

// 監聽 tab 切換，切換到 history 時跳轉
watch(activeTab, (val) => {
  if (val === 'history') {
    void router.push('/history');
  } else if (val === 'repair') {
    void router.push('/');
  }
});
</script>

<style scoped>
.upload-card {
  min-height: 400px;
}

.drag-area {
  border: 2px dashed #ccc;
  border-radius: 8px;
  background-color: #fafafa;
  transition: all 0.3s ease;
}

.drag-area:hover {
  border-color: #1976d2;
  background-color: #f0f8ff;
}

.preview-area {
  min-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  background-color: #fafafa;
}

.no-image {
  text-align: center;
}

.preview-image {
  max-width: 100%;
  max-height: 400px;
  object-fit: contain;
  border-radius: 4px;
}

.q-header {
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.canvas-container {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
}

.drawing-canvas {
  max-width: 100%;
  max-height: 400px;
  border-radius: 4px;
  cursor: crosshair;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.drawing-canvas:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.repair-success-mask {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(80, 80, 80, 0.3);
  z-index: 3;
  display: flex;
  align-items: center;
  justify-content: center;
}
.fade-success-leave-active {
  transition: opacity 1s;
}
.fade-success-leave-to {
  opacity: 0;
}
.fade-success-leave-from {
  opacity: 1;
}

/* 參考圖預覽尺寸 */
.ref-preview {
  width: 100%;
  max-width: 220px;
  border-radius: 8px;
  margin: 0 auto;
}
</style>
