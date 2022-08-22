import multiprocessing as mp
from pathlib import Path

import click
import more_itertools as mi
from PIL import Image
from returns.curry import partial


def convert_image(file_path: Path, output_dir: Path, img_size: int) -> None:
    image = Image.open(file_path)
    # convert image to gray scale
    image = image.convert("L")
    # resize image
    image = image.resize((img_size, img_size))
    # save image
    directory = output_dir / file_path.parent.name
    directory.mkdir(exist_ok=True)
    image.save(directory / file_path.name, optimize=True)


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
    default="resized",
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
def main(input_dir: Path, output_dir: Path, img_size: int) -> None:
    image_paths = input_dir.glob("*/*.png")
    callback = partial(convert_image, output_dir=output_dir, img_size=img_size)
    with mp.Pool() as pool:
        mi.consume(pool.imap(callback, image_paths))


if __name__ == "__main__":
    main()
