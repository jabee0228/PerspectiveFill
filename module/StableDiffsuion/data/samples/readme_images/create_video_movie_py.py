from moviepy.editor import *
import glob 


images_dir = "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint/logs/2023-02-08_custom_keyboard_training_different_samplerSAMESEEDNOTEMA/images/train/"


clip = ImageSequenceClip(glob.glob(images_dir + "samples2_gs*"), fps=1)

# Overlay the text clip on the first video clip
# video = CompositeVideoClip(clip)

# Write the result to a file (many options available !)
clip.write_videofile("test.webm")