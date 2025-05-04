import json
import tensorflow as tf
from data_pipeline import parse_pair
from utils import load_config

def evaluate():
    cfg = load_config()

    with open(cfg['labels'], 'r') as f:
        labels = json.load(f)

    G = tf.keras.models.load_model(cfg['models_dir'] + '/generator')
    D = tf.keras.models.load_model(cfg['models_dir'] + '/discriminator')

    correct = 0
    for entry in labels:
        (a, b), label = parse_pair(entry, (cfg['img_height'], cfg['img_width']))
        real_score = D(tf.expand_dims(b, 0))
        fake = G(tf.random.normal([1, cfg['latent_dim']]))
        fake_score = D(fake)
        if (real_score > fake_score) == bool(label):
            correct += 1

    acc = correct / len(labels)
    print("Selection accuracy:", acc)

if __name__ == "__main__":
    evaluate()
