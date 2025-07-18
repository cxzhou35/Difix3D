import argparse
import os

from diffusers.utils import load_image

from src.pipeline_difix import DifixPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Input image")
    parser.add_argument(
        "-r", "--ref_image", type=str, help="(Optional) Reference image"
    )
    parser.add_argument("-o", "--output", type=str, help="(Optional) Output image path")
    parser.add_argument("-p", "--prompt", type=str, help="(Optional) Text prompt")
    parser.add_argument("-is", "--inference_steps", type=int, default=1)
    parser.add_argument("-ts", "--timesteps", type=int, default=199)
    parser.add_argument("-gs", "--guidance_scale", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    input_image_path = args.input
    ref_image_path = args.ref_image
    output_image_path = args.output

    if output_image_path is None:
        output_image_name = (
            f"{os.path.basename(input_image_path).split('.')[0]}_difix3d.png"
        )
        output_image_path = os.path.join("outputs", output_image_name)
        if not os.path.exists(output_image_path):
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    if args.prompt is None:
        # default prompt
        prompt = "remove degradation and artifacts in the image"
    else:
        prompt = args.prompt

    # load Difix pipeline
    pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
    pipe.to("cuda")

    input_image = load_image(input_image_path)

    if ref_image_path is not None:
        ref_image = load_image(ref_image_path)
    else:
        ref_image = None

    output_image = pipe(
        prompt,
        image=input_image,
        ref_image=ref_image,
        num_inference_steps=args.inference_steps,
        timesteps=[args.timesteps],
        guidance_scale=args.guidance_scale,
    ).images[0]

    output_image.save(output_image_path)


if __name__ == "__main__":
    main()
