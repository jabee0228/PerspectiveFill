import os
from PIL import Image



def resize_imagea(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.resize((512, 512))
            img.save(image_path)  # 直接覆蓋原本的檔案
        print(f"圖片已成功調整為 512x512 並覆蓋原檔案：{image_path}")
    except Exception as e:
        print(f"發生錯誤：{e}")


def resize_images(input_folder, output_folder, size=(512, 512)):
    """
    Resize all images in the input_folder to the specified size and save them to the output_folder.

    :param input_folder: Path to the folder containing the original images.
    :param output_folder: Path to the folder where resized images will be saved.
    :param size: Tuple indicating the target size (width, height).
    """
    # 如果輸出資料夾不存在，則創建
    os.makedirs(output_folder, exist_ok=True)

    # 遍歷輸入資料夾的所有檔案
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        # 檢查是否為有效的圖片檔案
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # 打開圖片並調整大小
                with Image.open(file_path) as img:
                    img_resized = img.resize(size, Image.Resampling.LANCZOS)
                    # 儲存調整後的圖片到輸出資料夾
                    output_path = os.path.join(output_folder, filename)
                    img_resized.save(output_path)
                    print(f"Resized and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    input_folder = input("input:")
    output_folder = input("output:")
    x = int(input("x:"))
    y = int(input("y:"))
    size = (x, y)

    resize_images(input_folder, output_folder, size)

