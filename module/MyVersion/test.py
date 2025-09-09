import tensorflow as tf
import load_data
from model import MyModel
from easydict import EasyDict as edict
import yaml
import load_data
import matplotlib.pyplot as plt

def load_yml(path):
    with open(path, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
            print(config)
            return edict(config)
        except yaml.YAMLError as exc:
            print(exc)

def load_test_data(fName):
    fixed = tf.image.decode_png(tf.io.read_file('./MyData/test/origin/'+fName))
    fixed = tf.image.resize(fixed, [512, 512])
    mask = tf.image.decode_png(tf.io.read_file('./MyData/test/mask/'+fName))
    mask = tf.image.resize(mask, [512, 512])

    fixed = fixed / 127.5 - 1.0
    mask = load_data.convert_mask(mask)
    fixed = fixed*(1-mask) + mask
    fixed = tf.expand_dims(fixed, 0)
    mask = tf.expand_dims(mask, 0)
    return [fixed, mask]

def generate_and_save_images(model, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    for i in range(predictions.shape[0]):
        plt.subplot(1, 2, 1)
        plt.imshow(test_input[0][i] * 0.5 + 0.5)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(predictions[i] * 0.5 + 0.5)
        plt.axis('off')
        plt.show()

config = load_yml('MyVersion/config1.yml')
config.BATCH_SIZE = 1
model = MyModel("Mymodel", config)
dir_path = './model_weight/generator'
model.generator.load_weights(dir_path)
# load fixed image and mask
test_input = load_test_data('image.png')
generate_and_save_images(model.generator, test_input)