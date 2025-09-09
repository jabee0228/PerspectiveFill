import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-in', '--in_path', type=str, required=True)
parser.add_argument('-out', '--out_path', type=str, required=True)

args = parser.parse_args()
dir_path = args.in_path
out_path = args.out_path

os.makedirs(out_path, exist_ok=True)  # 確保主輸出資料夾存在
print(f'Create directory: {out_path}')

fList = sorted(os.listdir(dir_path))  # 直接在這裡 sort，比你的方式簡潔

for i, file in enumerate(fList):
    assert file.endswith('.jpg'), f'Only support jpg file, but got {file}'

    subdir = os.path.join(out_path, f'{i // 400}')
    os.makedirs(subdir, exist_ok=True)  # 確保子資料夾存在

    src = os.path.join(dir_path, file)
    dst = os.path.join(subdir, file)

    os.system(f'mv "{src}" "{dst}"')  # 用雙引號避免空格問題
print(f'Move -> {dst}')
print('Done!')
