import cv2, glob, random, os
import numpy as np
random.seed(42)

input_images_folder = "../inpainting_original_paper/"
output_images_folder = "../output_inpainting_original_paper/"

image_radixes = [".".join(x.split(".")[:-1]) for x in glob.glob(input_images_folder + "*.png") if not "mask" in x ] 

# random images
n_samples = 3
selected_images = random.sample(image_radixes, k=n_samples)
print(selected_images)
# CREATE TRIPLETS
triplets_to_show = [(x + ".png", x + "_mask.png", glob.glob(output_images_folder + os.path.basename(x).split(".")[-1] + "*.png")[0]) for x in selected_images]
# triplets_to_show = [glob.glob(output_images_folder + os.path.basename(x).split(".")[-1] + "*.png")[0] for x in selected_images]

print(triplets_to_show)

# Set the size of the canvas and the number of rows and columns
image_size = 256 # squared
num_rows = n_samples
num_cols = 3 # image, mask and reconstructed
canvas_width = image_size * num_cols
canvas_height = image_size * num_rows

canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

print(canvas.shape)

for row in range(num_rows):
    for col in range(num_cols):
        # Calculate the position where the image should be placed
        x = col * image_size
        y = row * image_size
        
        # Get the corresponding image from the list
        img = cv2.imread(triplets_to_show[row][col])
        
        # Resize the image to fit in the grid cell
        img = cv2.resize(img, (image_size, image_size))
        
        # Place the image on the canvas at the calculated position
        canvas[y:y+image_size, x:x+image_size, :] = cv2.addWeighted(canvas[y:y+image_size, x:x+image_size, :], 0.5, img, 0.5, 0)


cv2.imwrite("show_samples.jpg", canvas)