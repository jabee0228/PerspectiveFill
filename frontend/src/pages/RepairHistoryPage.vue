<template>
  <q-page class="q-pa-lg">
    <div class="text-h5 q-mb-lg">修補紀錄</div>
    <div class="column q-gutter-md">
      <q-card v-for="record in records" :key="record.id" class="row items-center q-pa-md">
        <img :src="record.img" alt="修補圖" style="width:120px;height:auto;border-radius:8px;object-fit:cover;" />
        <div class="q-ml-md col">
          <div class="text-subtitle1">{{ record.title }}</div>
          <div class="text-caption text-grey-6">{{ record.date }}</div>
        </div>
        <div class="q-ml-auto row q-gutter-sm">
          <q-btn color="primary" icon="download" label="下載" @click="download(record)" />
          <q-btn color="secondary" icon="refresh" label="再次修補" @click="repairAgain(record)" />
        </div>
      </q-card>
    </div>
  </q-page>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import repairImg from '../assets/05043726_7787106650-68040068_3739775613_raw_640x480.jpg';

type RepairRecord = {
  id: number;
  img: string;
  title: string;
  date: string;
};

const records = ref<RepairRecord[]>([
  {
    id: 1,
    img: repairImg,
    title: '修補紀錄 1',
    date: '2024-08-08'
  },
  {
    id: 2,
    img: repairImg,
    title: '修補紀錄 2',
    date: '2024-08-07'
  }
]);

const download = (record: RepairRecord) => {
  // 下載功能實現
  console.log('下載:', record.title);
};

const repairAgain = (record: RepairRecord) => {
  // 再次修補功能實現
  console.log('再次修補:', record.title);
};
</script>

<style scoped>
.q-card {
  min-height: 100px;
}
</style>
