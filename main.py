import math
from pathlib import Path

import click
import tensorflow as tf
from matplotlib import pyplot as plt
from returns.curry import partial
from tqdm import tqdm

from loss import discriminator_loss, generator_loss
from model import make_discriminator_model, make_generator_model


def process_path(file_path: tf.Tensor, img_size: int) -> tf.Tensor:
    file_contents = tf.io.read_file(file_path)
    img = tf.io.decode_png(file_contents, channels=3)
    grayscale_img = tf.image.rgb_to_grayscale(img)
    resized_image = tf.image.resize(grayscale_img, [img_size, img_size])
    normalized_image = (resized_image - 127.5) / 127.5
    return normalized_image


def generate_and_save_images(
    output_dir: Path, model: tf.keras.Model, epoch: int, test_input: tf.Tensor
) -> None:
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))
    batch_size, *_ = predictions.shape
    for i in range(batch_size):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")

    plt.savefig(output_dir / f"image_at_epoch_{epoch:04d}.png")


@click.command()
@click.option(
    "--input-dir",
    default="images",
    type=click.Path(
        exists=True,
        dir_okay=True,
        file_okay=False,
        readable=True,
        path_type=Path,
    ),
    help="Input directory of images to train on.",
    required=True,
)
@click.option(
    "--output-dir",
    default="output",
    type=click.Path(
        exists=True,
        dir_okay=True,
        file_okay=False,
        writable=True,
        path_type=Path,
    ),
    help="Output directory of trained model and artifacts.",
    required=True,
)
@click.option(
    "--img-size",
    default=28,
    type=int,
    help="Size of output images.",
    required=True,
)
@click.option(
    "--noise-dim",
    default=100,
    help="Dimension of noise vector.",
    required=True,
    type=int,
)
@click.option(
    "--batch-size", default=256, type=int, help="Batch size.", required=True
)
@click.option(
    "--epochs", default=100, type=int, help="Number of epochs.", required=True
)
def main(
    input_dir: Path,
    output_dir: Path,
    img_size: int,
    noise_dim: int,
    batch_size: int,
    epochs: int,
) -> None:
    file_pattern = str(input_dir / "*" / "*.png")
    filenames_ds = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    cardinality = tf.data.experimental.cardinality(filenames_ds)
    images_ds = filenames_ds.map(
        partial(process_path, img_size=img_size),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    batched_ds = images_ds.batch(batch_size).prefetch(
        tf.data.experimental.AUTOTUNE
    )

    generator = make_generator_model(img_size=img_size, noise_dim=noise_dim)
    discriminator = make_discriminator_model(img_size=img_size)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )

    num_examples_to_generate = 16
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    @tf.function
    def train_step(images: tf.Tensor) -> None:
        noise = tf.random.normal([batch_size, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(cross_entropy, fake_output)
            disc_loss = discriminator_loss(
                cross_entropy, real_output, fake_output
            )

            gradients_of_generator = gen_tape.gradient(
                gen_loss, generator.trainable_variables
            )
            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, discriminator.trainable_variables
            )

            generator_optimizer.apply_gradients(
                zip(gradients_of_generator, generator.trainable_variables)
            )
            discriminator_optimizer.apply_gradients(
                zip(
                    gradients_of_discriminator,
                    discriminator.trainable_variables,
                )
            )

    def train(dataset, epochs):
        generate_and_save_images(output_dir, generator, 0, seed)

        for epoch in range(epochs):
            for image_batch in tqdm(
                dataset,
                total=math.ceil(cardinality / batch_size),
                desc=f"Epoch {epoch + 1}",
            ):
                train_step(image_batch)

            # Produce images for the GIF as you go
            generate_and_save_images(output_dir, generator, epoch + 1, seed)

            checkpoint.save(file_prefix=str(output_dir / "ckpt"))

    train(batched_ds, epochs)


if __name__ == "__main__":
    main()
