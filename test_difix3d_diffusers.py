import argparse
import os
from glob import glob
from os.path import join

import imageio
import numpy as np
from diffusers.utils import load_image
from tqdm import tqdm

from src.pipeline_difix import DifixPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Input image")
    parser.add_argument(
        "-r", "--ref_image", type=str, help="(Optional) Reference image"
    )
    parser.add_argument(
        "-o", "--output", type=str, help="(Optional) Output path", default=None
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="(Optional) Text prompt",
        default="remove degradation and artifacts in the image",
    )
    parser.add_argument("-is", "--inference_steps", type=int, default=1)
    parser.add_argument("-ts", "--timesteps", type=int, default=199)
    parser.add_argument("-gs", "--guidance_scale", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()

    # load input images
    if os.path.isdir(args.input):
        input_image_paths = sorted(glob(join(args.input, "*.png")))
    else:
        input_image_paths = [args.input]

    # load reference images if provided
    if args.ref_image is not None:
        if os.path.isdir(args.ref_image):
            ref_image_paths = sorted(glob(join(args.ref_image, "*")))
        else:
            ref_image_paths = [args.ref_image]

        assert len(input_image_paths) == len(
            ref_image_paths
        ), "Number of input images and reference images should be the same"

    output_dir = ""
    if args.output is not None:
        if (
            args.output.endswith(".mp4")
            or args.output.endswith(".png")
            or args.output.endswith(".jpg")
        ):
            output_dir = join(os.path.dirname(args.output))
        else:
            output_dir = join(args.output)
    else:
        output_dir = join("outputs")
    output_img_dir = join(output_dir, "images")
    os.makedirs(output_img_dir, exist_ok=True)

    # load Difix pipeline
    pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
    pipe.to("cuda")

    # processing
    output_images = []
    for idx, input_image_path in enumerate(
        tqdm(
            input_image_paths,
            desc="Processing input images",
            total=len(input_image_paths),
        )
    ):
        input_image = load_image(input_image_path)
        ref_image = (
            load_image(ref_image_paths[idx]) if args.ref_image is not None else None
        )
        output_image = pipe(
            args.prompt,
            image=input_image,
            ref_image=ref_image,
            num_inference_steps=args.inference_steps,
            timesteps=[args.timesteps],
            guidance_scale=args.guidance_scale,
        ).images[0]

        output_images.append(output_image)

        # save image
        output_image.save(
            join(
                output_dir,
                os.path.basename(input_image_path),
            )
        )

    # save output videos
    if len(output_images) > 1:
        imageio.mimwrite(
            join(output_dir, "output_video.mp4"),
            np.stack(output_images, axis=0),
            fps=60,
        )


if __name__ == "__main__":
    main()
