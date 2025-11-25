import cv2, glob, random
import numpy as np

random.seed(42)

target_image = "../custom_inpainting/mouse_keyboard/desk_pc_mouse2.png"
mask_image = "../custom_inpainting/mouse_keyboard/desk_pc_mouse2_mask.png"

reconstruction_folder = "../custom_inpainting/mouse_keyboard_reconstructed/"
image_reconstruction = sorted([x for x in glob.glob(reconstruction_folder + "*.png")])

# TWO ROWS
##### ONE FOR INPUT AND MASK
##### SECOND FOR GENERATION PIPELINE

# Set the size of the canvas and the number of rows and columns
image_size = 256 # squared
num_rows = 2
num_cols = 4 # 2 inputs in the center, and second row filled
canvas_width = image_size * num_cols
canvas_height = image_size * num_rows

canvas = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)

# FILL FIRST ROW

start = image_size # start to fill middle positions
input_canvas = np.concatenate([cv2.resize(cv2.imread(x), (image_size, image_size)) for x in [target_image, mask_image]],axis=1)

canvas[0:image_size, start:start+(image_size*2), :3] = input_canvas
cv2.imwrite("training.jpg", canvas)

# FILL SECOND ROW

generate_canvas = np.concatenate([cv2.resize(cv2.imread(x), (canvas_width//len(image_reconstruction),image_size )) for x in image_reconstruction], axis=1)
# print(generate_canvas.shape)
# exit()
canvas[image_size:, 0:generate_canvas.shape[1] , :3] = generate_canvas
cv2.imwrite("training.jpg", canvas)