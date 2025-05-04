import os
import json
import tensorflow as tf
from data_pipeline import parse_pair
from utils import load_config

# Optional: pip install tqdm for progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x  # fallback: no progress bar

def evaluate():
    cfg = load_config()

    with open(cfg['labels'], 'r') as f:
        labels = json.load(f)

    G = tf.keras.models.load_model(os.path.join(cfg['models_dir'], 'generator.keras'))
    D = tf.keras.models.load_model(os.path.join(cfg['models_dir'], 'discriminator.keras'))

    correct = 0
    total = len(labels)
    img_size = (cfg['img_height'], cfg['img_width'])

    for i, entry in enumerate(tqdm(labels, desc="Evaluating")):
        path_a = os.path.join(cfg['synth_dir'], entry['frame_a'])
        path_b = os.path.join(cfg['synth_dir'], entry['frame_b'])
        label = entry['label']

        # Load and preprocess images
        (a, b), _ = parse_pair(path_a, path_b, label, img_size)

        # Run inference
        real_score = D(tf.expand_dims(b, 0), training=False)
        fake = G(tf.random.normal([1, cfg['latent_dim']]), training=False)
        fake_score = D(fake, training=False)

        # Compare decision to label
        if (real_score > fake_score) == bool(label):
            correct += 1

    acc = correct / total
    print(f"\n✅ Selection accuracy: {acc:.4f}")

if __name__ == "__main__":
    evaluate()
