import os
import tensorflow as tf
from data_pipeline import make_dataset, load_labels
from models import build_generator, build_discriminator
from utils import load_config, ensure_dir

def train():
    cfg = load_config()
    ensure_dir(cfg['models_dir'])
    ensure_dir(cfg['outputs_dir'])

    img_size = (cfg['img_height'], cfg['img_width'])
    entries = load_labels(cfg['labels'])  # used to calculate steps per epoch
    ds = make_dataset(cfg['labels'], cfg['batch_size'], img_size)
    steps_per_epoch = len(entries) // cfg['batch_size']

    G = build_generator(cfg['latent_dim'], *img_size, channels=3)
    D = build_discriminator((*img_size, 3))

    g_opt = tf.keras.optimizers.Adam(cfg['learning_rate'], beta_1=cfg['beta_1'])
    d_opt = tf.keras.optimizers.Adam(cfg['learning_rate'], beta_1=cfg['beta_1'])
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    for epoch in range(cfg['epochs']):
        print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")
        for step, ((a, b), label) in enumerate(ds):
            if step >= steps_per_epoch:
                break  # ensures the dataset does not hang due to repeat-less behavior

            # D-step
            with tf.GradientTape() as td:
                fake = G(tf.random.normal([tf.shape(a)[0], cfg['latent_dim']]))
                real_logits = D(b)
                fake_logits = D(fake)
                d_loss = (
                    bce(tf.ones_like(real_logits), real_logits) +
                    bce(tf.zeros_like(fake_logits), fake_logits)
                )
            grads = td.gradient(d_loss, D.trainable_variables)
            d_opt.apply_gradients(zip(grads, D.trainable_variables))

            # G-step
            with tf.GradientTape() as tg:
                fake = G(tf.random.normal([tf.shape(a)[0], cfg['latent_dim']]))
                fake_logits = D(fake)
                g_loss = bce(tf.ones_like(fake_logits), fake_logits)
            grads = tg.gradient(g_loss, G.trainable_variables)
            g_opt.apply_gradients(zip(grads, G.trainable_variables))

        print(f"Epoch {epoch+1}/{cfg['epochs']} — D: {d_loss:.4f}, G: {g_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            G.save(os.path.join(cfg['models_dir'], 'generator.keras'))
            D.save(os.path.join(cfg['models_dir'], 'discriminator.keras'))


if __name__ == "__main__":
    train()
