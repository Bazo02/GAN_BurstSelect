import json
import os
import tensorflow as tf
from utils import normalize_img, load_config

# Laster base_dir fra config for syntetiske bilder
_config = load_config()
BASE_DIR = _config['synth_dir']

# Leser labels fra JSON-fil
def load_labels(path):
    with open(path, 'r') as f:
        return json.load(f)

# Parser én datapunkt: full path til skarpt og degradert bilde, samt label
def parse_pair(path_a, path_b, label, img_size):
    img_a = tf.io.decode_jpeg(tf.io.read_file(path_a), channels=3)
    img_b = tf.io.decode_jpeg(tf.io.read_file(path_b), channels=3)
    img_a = normalize_img(img_a, img_size)
    img_b = normalize_img(img_b, img_size)
    return (img_a, img_b), tf.cast(label, tf.float32)

# Lager tf.data.Dataset fra labels-json, bruker BASE_DIR internt
# Signature matcher train_gan: labels_json, batch_size, img_size

def make_dataset(labels_json, batch_size, img_size):
    entries = load_labels(labels_json)
    # Bygg fullstendig sti basert på BASE_DIR
    frame_as = [os.path.join(BASE_DIR, e['frame_a']) for e in entries]
    frame_bs = [os.path.join(BASE_DIR, e['frame_b']) for e in entries]
    labels   = [e['label'] for e in entries]

    ds = tf.data.Dataset.from_tensor_slices((frame_as, frame_bs, labels))
    ds = ds.map(lambda a, b, l: parse_pair(a, b, l, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds