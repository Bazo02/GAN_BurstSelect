import tensorflow as tf
import yaml
import os

def normalize_img(img, img_size):
    # endrer størrelse og normaliserer fra [0,255] til [-1,1]
    img = tf.image.resize(img, img_size)
    return (tf.cast(img, tf.float32) / 127.5) - 1.0

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    # oppretter katalog hvis ikke eksisterer
    os.makedirs(path, exist_ok=True)


