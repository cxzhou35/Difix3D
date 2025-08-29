import argparse
import os
from glob import glob

import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

from model import Difix


# 32 - 62 long-focus views
REF_VIEW_DICT = {
    "32": "06",
    "33": "04",
    "34": "02",
    "35": "02",
    "36": "02",
    "37": "04",
    "38": "04",
    "39": "04",
    "40": "02",
    "41": "04",
    "42": "04",
    "43": "06",
    "44": "06",
    "45": "06",
    "46": "06",
    "47": "08",
    "48": "08",
    "49": "08",
    "50": "10",
    "51": "12",
    "52": "12",
    "53": "00",
    "54": "00",
    "55": "12",
    "56": "12",
    "57": "14",
    "58": "12",
    "59": "14",
    "60": "14",
    "61": "10",
    "62": "08",
}

def parse_args():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input image or directory",
    )
    parser.add_argument(
        "--ref_path",
        type=str,
        default=None,
        help="Path to the reference image or directory",
    )
    parser.add_argument(
        "--height", type=int, default=900, help="Height of the input image"
    )
    parser.add_argument(
        "--width", type=int, default=1600, help="Width of the input image"
    )
    parser.add_argument(
        "--data_format", type=str, default="general", help="Save format of the data (evc, general)"
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="The prompt to be used"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the pretrained model to be used",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a model state dict to be used",
    )
    parser.add_argument(
        "--output_path", type=str, default="outputs", help="Path to save the output"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed to be used")
    parser.add_argument("--timestep", type=int, default=199, help="Diffusion timestep")
    parser.add_argument("--video", action="store_true", help="If the input is a video")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite the output")
    args = parser.parse_args()

    return args


def get_evc_meta(img_path: str):
    # format: xx_xxxxxx.png
    view_id, frame_id = img_path.split("/")[-1].split(".")[0].split("_")
    return view_id, frame_id


def save_evc_img_path(img, output_path: str, view_id: str, frame_id: str):
    assert os.path.exists(output_path), "Save path does not exist"
    view_save_dir = os.path.join(output_path, "images", f"{int(view_id):02d}")
    os.makedirs(view_save_dir, exist_ok=True)
    return os.path.join(view_save_dir, f"{int(frame_id):06d}.png")


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # Initialize the model
    model = Difix(
        pretrained_name=args.model_name,
        pretrained_path=args.model_path,
        timestep=args.timestep,
        mv_unet=True if args.ref_path is not None else False,
    )
    model.set_eval()

    # Load input images
    if os.path.isdir(args.input_path):
        input_images = sorted(glob(os.path.join(args.input_path, "*.png")))
    else:
        input_images = [args.input_path]

    # Load reference images if provided
    if args.ref_path is not None and args.data_format != "evc":
        if os.path.isdir(args.ref_path):
            ref_images = sorted(glob(os.path.join(args.ref_path, "*.png")))
        else:
            ref_images = [args.ref_path]
        assert len(input_images) == len(
            ref_images
        ), "Number of input images and reference images should be the same"

    pred_images = []
    for idx, input_image in enumerate(tqdm(input_images, desc="Processing images")):
        image = Image.open(input_image).convert("RGB")

        if args.data_format == "evc":
            # HACK: get the reference image based on the view id
            view_id, frame_id = get_evc_meta(input_image)
            ref_view_id = REF_VIEW_DICT[view_id]
            ref_image_path = os.path.join(args.ref_path, f"{int(ref_view_id):02d}_{int(frame_id):06d}.png")
            output_image_path = save_evc_img_path(image, args.output_path, view_id, frame_id)
        else:
            ref_image_path = ref_images[idx]
            output_image_dir = os.path.join(args.output_path, "images")
            output_image_path = os.path.join(output_image_dir, os.path.basename(input_image))

        ref_image = (
            Image.open(ref_image_path).convert("RGB")
            if args.ref_path is not None
            else None
        )

        # Skip if the output image already exists
        if os.path.exists(output_image_path) and not args.force:
            print(f"Skipping {output_image_path}, it already exists")
            continue

        pred_image = model.sample(
            image,
            height=args.height,
            width=args.width,
            ref_image=ref_image,
            prompt=args.prompt,
        )
        pred_images.append(pred_image)
        pred_image.save(output_image_path)

    # Save outputs
    if args.video:
        # Save as video
        video_path = os.path.join(args.output_path, "output.mp4")
        writer = imageio.get_writer(video_path, fps=30)
        for pred_image in tqdm(pred_images, desc="Saving video"):
            writer.append_data(np.array(pred_image))
        writer.close()


if __name__ == "__main__":
    main()
