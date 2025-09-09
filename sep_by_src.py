import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('-in', '--in_path', type=str, required=True)
parser.add_argument('-out', '--out_path', type=str, required=True)
args = parser.parse_args()
dir_path = args.in_path
out_path = args.out_path


fList = sorted(os.listdir(dir_path))
dirs = {}
for f in fList:
    assert f.endswith('.jpg'), f'File {f} is not a jpg file'
    src_img = f[:-10] # remove '_fixed.jpg'
    src_img = src_img.split('-')[0]
    dirs.setdefault(src_img, []).append(f)

os.makedirs(out_path, exist_ok=True)  # 確保主輸出資料夾存在
print(f'Create directory: {out_path}')

for src_img, files in dirs.items():
    os.makedirs(os.path.join(out_path, src_img), exist_ok=True)  # 確保每個子資料夾存在
    print(f'Create directory: {os.path.join(out_path, src_img)}')
    for f in files:
        src_path = os.path.join(dir_path, f)
        dst_path = os.path.join(out_path, src_img, f)
        shutil.move(src_path, dst_path)
print('Done')
