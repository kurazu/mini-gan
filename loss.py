import tensorflow as tf


def discriminator_loss(
    cross_entropy, real_output: tf.Tensor, fake_output: tf.Tensor
) -> tf.Tensor:

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(cross_entropy, fake_output: tf.Tensor) -> tf.Tensor:
    return cross_entropy(tf.ones_like(fake_output), fake_output)
