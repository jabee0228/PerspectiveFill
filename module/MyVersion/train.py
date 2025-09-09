import yaml
from easydict import EasyDict as edict
import tensorflow as tf
from .model import MyModel
from .trainer import Trainer
from .load_data import create_dataset

def load_yml(path):
    with open(path, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
            print(config)
            return edict(config)
        except yaml.YAMLError as exc:
            print(exc)






if __name__ == "__main__":
    # load config.yml file
    config = load_yml('MyVersion/config1.yml')
    if config.GPU_ID != -1:
        gpu_ids = config.GPU_ID
    else:
        gpu_ids = [0]

    print('loading train&validation data...')

    '''    # load training data
    data_dir = config.TRAIN_LIST
    data_dir = './data/examples/train/'
    train_ds =  tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None,
        validation_split=0.2,
        subset="training",
        seed=123,
        shuffle=True,
        image_size=(512, 512),
        batch_size=config.BATCH_SIZE,
        color_mode="rgb"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None,
        validation_split=0.2,
        subset="validation",
        seed=123,
        shuffle=True,
        image_size=(512, 512),
        batch_size=config.BATCH_SIZE,
        color_mode="rgb"
    )'''


    # get dataset
    train_ds = create_dataset(config.TRAIN_PATH, config.IMG_SHAPE, config.BATCH_SIZE)
    print('train_ds', train_ds.element_spec)

    model = MyModel("Mymodel", config)
    # model.generator.build(input_shape=(config.BATCH_SIZE, 512, 512, 3))
    # model.generator.summary()
    # model.discriminator.summary()
    # def weight path
    dir_path = './model_weight'
    log_path = './train_log'
    # trainer = Trainer(model, config, dir_path)
    trainer = Trainer(model, config)
    trainer.train(train_ds, epochs=config.MAX_ITERS, dir_path=dir_path, log_path=log_path,continue_training=config.CONTINUE_TRAIN)
    # batch size 調整
    # 同一地點的圖片 training
    # 用相同做test
    # 印出 discrimnator accuracy