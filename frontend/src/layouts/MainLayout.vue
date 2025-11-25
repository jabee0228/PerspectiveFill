<template>
  <q-layout view="lHh Lpr lFf">
    <!-- 頂部導航欄 -->
    <q-header elevated class="bg-primary text-white">
      <q-toolbar>
        <!-- 網站名稱 -->
        <q-toolbar-title class="text-h5 text-weight-bold">
          PerspectiveFill
        </q-toolbar-title>
      </q-toolbar>
    </q-header>

    <q-page-container>
      <router-view />
    </q-page-container>
  </q-layout>
</template>

<script setup lang="ts">
// 無需任何腳本：已移除 activeTab 與設定按鈕邏輯
</script>
