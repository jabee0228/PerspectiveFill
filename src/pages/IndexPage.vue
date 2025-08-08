<template>
  <q-page>
    <!-- 頂部導航欄 -->
    <q-header elevated class="bg-primary text-white">
      <q-toolbar>
        <!-- 網站名稱 -->
        <q-toolbar-title class="text-h5 text-weight-bold">
          PerspectiveFill
        </q-toolbar-title>

        <!-- 導航菜單 -->
        <q-tabs v-model="activeTab" class="q-ml-lg">
          <q-tab name="repair" label="修補" />
          <q-tab name="history" label="修補紀錄" />
        </q-tabs>

        <!-- 設定按鈕 -->
        <q-btn
          flat
          round
          icon="settings"
          class="q-ml-auto"
          @click="openSettings"
        >
          <q-tooltip>設定</q-tooltip>
        </q-btn>
      </q-toolbar>
    </q-header>

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
          <q-card class="reference-card q-mt-md" elevated>
            <q-card-section class="text-center">
              <div class="text-h6 q-mb-md">參考圖選擇</div>
              <q-select
                v-model="referenceMode"
                :options="referenceOptions"
                label="選擇參考圖來源"
                outlined
                dense
                class="full-width"
                emit-value
                map-options
              />
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

                <div v-else class="canvas-container" style="position: relative;">
                  <canvas
                    ref="canvas"
                    class="drawing-canvas"
                    @mousedown="startDrawing"
                    @mousemove="draw"
                    @mouseup="stopDrawing"
                    @mouseleave="stopDrawing"
                  />
                  <!-- 修補中灰色遮罩 -->
                  <div v-if="isRepairing" style="position:absolute;top:0;left:0;width:100%;height:100%;background:rgba(80,80,80,0.5);z-index:2;display:flex;align-items:center;justify-content:center;">
                    <div style="width:80%;position:absolute;bottom:24px;left:10%;">
                      <q-linear-progress :value="repairProgress/100" color="primary" track-color="grey-4" rounded size="20px" />
                    </div>
                  </div>
                  <!-- 修補成功提示 -->
                  <transition name="fade-success">
                    <div v-if="repairSuccess" class="repair-success-mask">
                      <div class="bg-white text-primary text-h5 q-pa-xl q-mb-xl repair-success-tip">修補成功！</div>
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
import { ref, nextTick } from 'vue';
import repairResultImg from '../assets/05043726_7787106650-68040068_3739775613_raw_640x480.jpg';

const activeTab = ref('repair');
const selectedImage = ref<string | null>(null);
const canvas = ref<HTMLCanvasElement | null>(null);
const isDrawing = ref(false);
const brushSize = ref(20);
const referenceMode = ref('custom');
const isRepairing = ref(false);
const repairProgress = ref(0);
const repairSuccess = ref(false);

let ctx: CanvasRenderingContext2D | null = null;

const referenceOptions = [
  { label: '自訂義上傳', value: 'custom' },
  { label: 'Google Places API', value: 'google' }
];

const openSettings = () => {
  // 開啟設定對話框
  console.log('開啟設定');
};

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

const setupCanvas = () => {
  if (!canvas.value || !selectedImage.value) return;

  const img = new Image();
  img.onload = () => {
    const canvasEl = canvas.value!;
    ctx = canvasEl.getContext('2d')!;

    // 設定 canvas 尺寸
    const containerWidth = canvasEl.parentElement!.clientWidth;
    const scale = Math.min(containerWidth / img.width, 400 / img.height);

    canvasEl.width = img.width * scale;
    canvasEl.height = img.height * scale;

    // 繪製圖片
    ctx.drawImage(img, 0, 0, canvasEl.width, canvasEl.height);
  };
  img.src = selectedImage.value;
};

const startDrawing = (event: MouseEvent) => {
  if (!ctx) return;

  isDrawing.value = true;

  // 設定紅色畫筆，透明度 0.1，且不會疊加顏色深度
  ctx.globalCompositeOperation = 'source-over';
  ctx.strokeStyle = 'rgba(255,0,0,0.1)';
  ctx.lineWidth = brushSize.value;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  const rect = canvas.value!.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  ctx.beginPath();
  ctx.moveTo(x, y);
};

const draw = (event: MouseEvent) => {
  if (!isDrawing.value || !ctx) return;

  const rect = canvas.value!.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  ctx.lineTo(x, y);
  ctx.stroke();
};

const stopDrawing = () => {
  if (!isDrawing.value) return;

  isDrawing.value = false;
  if (ctx) {
    ctx.beginPath();
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
};

const startRepair = () => {
  isRepairing.value = true;
  repairSuccess.value = false;
  repairProgress.value = 0;

  const interval = setInterval(() => {
    if (repairProgress.value < 100) {
      repairProgress.value += 2;
    } else {
      clearInterval(interval);
      repairSuccess.value = true;
      isRepairing.value = false;
      selectedImage.value = repairResultImg;
      void nextTick(() => {
        setupCanvas();
      });
      // 2秒後淡出修補成功提示
      setTimeout(() => {
        repairSuccess.value = false;
      }, 2000);
    }
  }, 300);
};
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
  top: 0; left: 0; width: 100%; height: 100%;
  background: rgba(80,80,80,0.3);
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
</style>
