import cv2
import tensorflow as tf
from .model import MyModel
from easydict import EasyDict as edict
import yaml
from module.MyVersion import load_data

import matplotlib.pyplot as plt

def load_yml(path):
    with open(path, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
            # print(config)
            return edict(config)
        except yaml.YAMLError as exc:
            print(exc)
def loadBatchData(loftrDir, maskDir, batch_size=4):
    loftr_images = []
    mask_images = []

    loftr_files = sorted(tf.io.gfile.glob(f'{loftrDir}/*.png'))
    mask_files = sorted(tf.io.gfile.glob(f'{maskDir}/*.png'))

    for loftr_file, mask_file in zip(loftr_files, mask_files):
        loftr_image = tf.image.decode_png(tf.io.read_file(loftr_file))
        loftr_image = tf.image.resize(loftr_image, [512, 512])
        loftr_image = loftr_image / 127.5 - 1.0
        loftr_images.append(loftr_image)

        mask_image = tf.image.decode_png(tf.io.read_file(mask_file))
        mask_image = tf.image.resize(mask_image, [512, 512])
        mask_image = load_data.convert_mask(mask_image)
        mask_images.append(mask_image)

    loftr_images = tf.stack(loftr_images)
    mask_images = tf.stack(mask_images)

    dataset = tf.data.Dataset.from_tensor_slices((loftr_images, mask_images))
    dataset = dataset.batch(batch_size)
    return dataset

def load_test_data(loftrDir, maskDir):
    fixed = tf.image.decode_png(tf.io.read_file(loftrDir))
    fixed = tf.image.resize(fixed, [512, 512])
    mask = tf.image.decode_png(tf.io.read_file(maskDir))
    mask = tf.image.resize(mask, [512, 512])

    fixed = fixed / 127.5 - 1.0
    mask = load_data.convert_mask(mask)
    fixed = tf.expand_dims(fixed, 0)
    mask = tf.expand_dims(mask, 0)
    return [fixed, mask]

class InferModel:
    def __init__(self, config, weightDir='./module/model_weight/generator'):
        self.model = MyModel("Mymodel", config)
        weightDir = './module/model_weight/generator'
        self.model.generator.load_weights(weightDir)

    def inference_batch(self, inputs, batch_size):
        """
        Generate images for a batch of loftr images and mask images.

        :param  loftrDir: directory of input fixed images
        :param maskDir: directory of mask images

        return: (predictions)
        """
        data = [inputs['original_images'], inputs['masks']]
        predictions = self.model.generator(data, training=False)
        return predictions, inputs['original_images']


def generate_and_save_images(model, test_input, newName, outputDir='./imagesoutput/'):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    predictions = tf.split(predictions, predictions.get_shape().as_list()[0], axis=0) # split batch
    img = (predictions[0][0] * 0.5 + 0.5) * 255
    img = tf.cast(img, tf.uint8)
    img = tf.image.encode_png(img)
    tf.io.write_file(outputDir+newName, img)

def inference(loftrDir, maskDir, newName):
    # load config
    config = load_yml('./module/MyVersion/config1.yml')
    config.BATCH_SIZE = 1
    model = MyModel("Mymodel", config)
    dir_path = './module/model_weight/generator' # model weight path
    model.generator.load_weights(dir_path)
    # load fixed image and mask
    test_input = load_test_data(loftrDir, maskDir)
    generate_and_save_images(model.generator, test_input, newName)

