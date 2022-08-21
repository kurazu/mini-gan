import tensorflow as tf


def make_generator_model(*, img_size: int, noise_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(
            7 * 7 * 256, use_bias=False, input_shape=(noise_dim,)
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (
        None,
        7,
        7,
        256,
    )  # Note: None is the batch size

    model.add(
        tf.keras.layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 7, 7, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(
        tf.keras.layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 14, 14, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(
        tf.keras.layers.Conv2DTranspose(
            1,
            (5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            activation="tanh",
        )
    )
    assert model.output_shape == (None, img_size, img_size, 1)

    model.summary()
    return model


def make_discriminator_model(*, img_size: int) -> tf.keras.Model:
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            64,
            (5, 5),
            strides=(2, 2),
            padding="same",
            input_shape=[img_size, img_size, 1],
        )
    )
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(
        tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same")
    )
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model
