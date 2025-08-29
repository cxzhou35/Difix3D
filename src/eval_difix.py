import argparse
import gc
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import RandomSampler
import torchvision
import json
import lpips
from einops import rearrange
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from dataset import PairedDataset
from model import Difix

p = "src/"
sys.path.append(p)
from loss import psnr, ssim, mse

def set_env_args():
    # set proxy mirrors
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    # os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["OVEN_HOME"] = "/root/.config/oven"


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset options
    parser.add_argument("--root_dir", required=True, type=str)
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--prompt", default=None, type=str)
    parser.add_argument("--image_width", default=1024, type=int)
    parser.add_argument("--image_height", default=576, type=int)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])

    # validation eval args
    parser.add_argument(
        "--num_samples_eval",
        type=int,
        default=100,
        help="Number of samples to use for all evaluation",
    )

    # model args
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--mv_unet", action="store_true")
    parser.add_argument("--timestep", default=199, type=int)

    # eval details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
    )
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    return args


def main(args):
    # load model
    net_difix = Difix(
        pretrained_path=args.pretrained_path,
        timestep=args.timestep,
        mv_unet=args.mv_unet,
    )
    net_difix.set_eval()

    # make train/test dataloaders
    dataset_train = PairedDataset(
        root_dir=args.root_dir, dataset_path=args.dataset_path, split="train", tokenizer=net_difix.tokenizer, height=args.image_height, width=args.image_width
    )
    train_sampler = RandomSampler(
        dataset_train, replacement=True, num_samples=1200,
    )
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        sampler=train_sampler,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    dataset_val = PairedDataset(
        root_dir=args.root_dir, dataset_path=args.dataset_path, split="test", tokenizer=net_difix.tokenizer, height=args.image_height, width=args.image_width
    )
    dl_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=args.dataloader_num_workers
    )

    # metrics
    net_lpips = lpips.LPIPS(net="vgg").cuda()
    net_lpips.requires_grad_(False)
    net_vgg = torchvision.models.vgg16(pretrained=True).features
    for param in net_vgg.parameters():
        param.requires_grad_(False)


    # Move all networks to device and cast to weight_dtype
    net_difix.to(args.device)
    net_lpips.to(args.device)
    net_vgg.to(args.device)

    # renorm with image net statistics
    t_vgg_renorm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # start the evaluate loop
    dl = dl_train if args.split == "train" else dl_val
    logs = {
        "mse": [],
        "lpips": [],
        "psnr": [],
        "ssim": [],
    }
    for step, batch in enumerate(tqdm(dl, desc=f"Evaluating {args.split} dataset")):
        with torch.no_grad():
            x_src = batch["conditioning_pixel_values"].to(args.device)  # input image (condition)
            x_tgt = batch["output_pixel_values"].to(args.device)  # target image (gt)
            data_id = batch["data_id"]
            B, V, C, H, W = x_src.shape

            # forward pass
            x_tgt_pred = net_difix(
                x_src, prompt_tokens=batch["input_ids"].to(args.device)
            )  # model predicted image

            x_tgt = x_tgt[:, 0]
            x_tgt_pred = x_tgt_pred[:, 0]
            x_src = x_src[:, 0]

            # metrics
            mse = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean")
            lpips_value = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean()

            x_tgt = torch.clamp((x_tgt + 1) / 2., 0, 1.)
            x_tgt_pred = torch.clamp((x_tgt_pred + 1) / 2., 0, 1.)
            x_src = torch.clamp((x_src + 1) / 2., 0, 1.)

            psnr_value = psnr(x_tgt_pred.float(), x_tgt.float()).mean()
            ssim_value = ssim(x_tgt_pred.float(), x_tgt.float()).mean()

            logs["mse"].append(mse.item())
            logs["lpips"].append(lpips_value.item())
            logs["psnr"].append(psnr_value.item())
            logs["ssim"].append(ssim_value.item())

            # save images
            output_image_dir = os.path.join(args.output_dir, args.split, "images")
            os.makedirs(output_image_dir, exist_ok=True)

            tgt_image = rearrange(x_tgt, "b c h w -> (b h) w c").float().detach().cpu()
            pred_image = rearrange(x_tgt_pred, "b c h w -> (b h) w c").float().detach().cpu()
            src_image = rearrange(x_src, "b c h w -> (b h) w c").float().detach().cpu()

            concat_image = np.concatenate([src_image, pred_image, tgt_image], axis=1)
            concat_image = (concat_image * 255).astype(np.uint8)
            Image.fromarray(concat_image).save(os.path.join(output_image_dir, f"{data_id[0]}.png"))

    logs["avg_mse"] = np.mean(logs["mse"]).item()
    logs["avg_lpips"] = np.mean(logs["lpips"]).item()
    logs["avg_psnr"] = np.mean(logs["psnr"]).item()
    logs["avg_ssim"] = np.mean(logs["ssim"]).item()


    # save logs
    with open(os.path.join(args.output_dir, args.split, f"{args.split}_metrics.json"), "w") as f:
        json.dump(logs, f, indent=4)

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()

    set_env_args()

    # save args
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    main(args)
