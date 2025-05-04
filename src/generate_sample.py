import tensorflow as tf
import os
from utils import load_config
from tensorflow.keras.utils import save_img

# Last inn config og modellen
cfg = load_config()
G = tf.keras.models.load_model(os.path.join(cfg['models_dir'], 'generator.keras'))

# Sørg for at output-mappen eksisterer
os.makedirs(cfg['outputs_dir'], exist_ok=True)

# Generer 5 bilder
for i in range(5):
    noise = tf.random.normal([1, cfg['latent_dim']])
    fake_img = G(noise, training=False)[0]

    # Reskaler fra [-1, 1] til [0, 1]
    fake_img = (fake_img + 1.0) / 2.0
    save_path = os.path.join(cfg['outputs_dir'], f"fake_sample_{i+1}.png")
    save_img(save_path, fake_img.numpy())
    print(f"✅ Lagret: {save_path}")
