import tensorflow as tf

def build_discriminator(input_shape):
    base = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=input_shape,
        pooling='avg',
        weights='imagenet'
    )
    x = tf.keras.layers.Dense(1)(base.output)
    return tf.keras.Model(base.input, x, name='Discriminator')

def build_generator(latent_dim, img_height, img_width, channels=3):
    inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(8*8*256, use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Reshape((8, 8, 256))(x)
    for filters in [128, 64, 32, 16]:
        x = tf.keras.layers.Conv2DTranspose(filters, 5, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    outputs = tf.keras.layers.Conv2DTranspose(
        channels, 5, strides=1, padding='same', activation='tanh'
    )(x)
    return tf.keras.Model(inputs, outputs, name='Generator')
