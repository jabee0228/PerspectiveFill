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
      <div class="row q-gutter-lg">
        <!-- 左側上傳區域 -->
        <div class="col-12 col-md-4">
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
        </div>

        <!-- 右側預覽和操作區域 -->
        <div class="col-12 col-md-8">
          <q-card elevated>
            <q-card-section>
              <div class="text-h6 q-mb-md">圖片預覽</div>

              <!-- 圖片預覽區域 -->
              <div class="preview-area">
                <div v-if="!selectedImage" class="no-image">
                  <q-icon name="image" size="5rem" color="grey-4" />
                  <div class="text-body1 text-grey-6 q-mt-md">
                    請選擇要修補的圖片
                  </div>
                </div>

                <img
                  v-else
                  :src="selectedImage"
                  alt="預覽圖片"
                  class="preview-image"
                />
              </div>
            </q-card-section>

            <!-- 操作按鈕 -->
            <q-card-actions v-if="selectedImage" align="right">
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
import { ref } from 'vue';

const activeTab = ref('repair');
const selectedImage = ref<string | null>(null);

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
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        selectedImage.value = e.target?.result as string;
      };
      reader.readAsDataURL(file);
    }
  }
};

const clearImage = () => {
  selectedImage.value = null;
};

const startRepair = () => {
  // 開始圖片修補處理
  console.log('開始修補圖片');
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
</style>
