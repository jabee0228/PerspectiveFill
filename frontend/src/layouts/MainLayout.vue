<template>
  <q-layout view="lHh Lpr lFf">
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

    <q-page-container>
      <router-view />
    </q-page-container>
  </q-layout>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

const route = useRoute()
const router = useRouter()
const activeTab = ref('repair')

// 根據當前路由設置活動標籤
watch(() => route.path, (newPath) => {
  if (newPath === '/') {
    activeTab.value = 'repair'
  } else if (newPath === '/repair-history') {
    activeTab.value = 'history'
  }
}, { immediate: true })

// 監聽 tab 切換
watch(activeTab, (val) => {
  if (val === 'history') {
    void router.push('/repair-history')
  } else if (val === 'repair') {
    void router.push('/')
  }
})

const openSettings = () => {
  // 開啟設定對話框
  console.log('開啟設定')
}
</script>
