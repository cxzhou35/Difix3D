import os
import sys

import numpy as np
import requests
import torch
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler
from peft import LoraConfig
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel

p = "src/"
sys.path.append(p)
from einops import rearrange, repeat


def make_1step_sched():
    noise_scheduler_1step = DDPMScheduler.from_pretrained(
        "stabilityai/sd-turbo", subfolder="scheduler"
    )
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        # note: save the features for skip connection in the decoder
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    # post-process
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [
            self.skip_conv_1,
            self.skip_conv_2,
            self.skip_conv_3,
            self.skip_conv_4,
        ]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


def download_url(url, outf):
    if not os.path.exists(outf):
        print(f"Downloading checkpoint to {outf}")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(outf, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        print(f"Downloaded successfully to {outf}")
    else:
        print(f"Skipping download, {outf} already exists")


def load_ckpt_from_state_dict(net_difix, optimizer, pretrained_path):
    sd = torch.load(pretrained_path, map_location="cpu")

    if "state_dict_vae" in sd:
        _sd_vae = net_difix.vae.state_dict()
        for k in sd["state_dict_vae"]:
            _sd_vae[k] = sd["state_dict_vae"][k]
        net_difix.vae.load_state_dict(_sd_vae)
    _sd_unet = net_difix.unet.state_dict()
    for k in sd["state_dict_unet"]:
        _sd_unet[k] = sd["state_dict_unet"][k]
    net_difix.unet.load_state_dict(_sd_unet)

    optimizer.load_state_dict(sd["optimizer"])

    return net_difix, optimizer


def save_ckpt(net_difix, optimizer, outf):
    sd = {}
    sd["vae_lora_target_modules"] = net_difix.target_modules_vae
    sd["rank_vae"] = net_difix.lora_rank_vae
    sd["state_dict_unet"] = net_difix.unet.state_dict()
    sd["state_dict_vae"] = {
        k: v
        for k, v in net_difix.vae.state_dict().items()
        if "lora" in k or "skip" in k
    }

    sd["optimizer"] = optimizer.state_dict()

    torch.save(sd, outf)


class Difix(torch.nn.Module):
    def __init__(
        self,
        pretrained_name=None,
        pretrained_path=None,
        ckpt_folder="checkpoints",
        lora_rank_vae=4,
        mv_unet=False,
        timestep=999,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "stabilityai/sd-turbo", subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "stabilityai/sd-turbo", subfolder="text_encoder"
        ).cuda()

        # note: set the scheduler for one step denoising
        self.sched = make_1step_sched()

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")

        # note: override the forward pass for the vae encoder and decoder
        vae.encoder.forward = my_vae_encoder_fwd.__get__(
            vae.encoder, vae.encoder.__class__
        )
        vae.decoder.forward = my_vae_decoder_fwd.__get__(
            vae.decoder, vae.decoder.__class__
        )
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(
            512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(
            256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(
            128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).cuda()
        vae.decoder.ignore_skip = False

        if mv_unet:
            from mv_unet import UNet2DConditionModel
        else:
            from diffusers import UNet2DConditionModel

        # todo: check how to finetune the unet
        unet = UNet2DConditionModel.from_pretrained(
            "stabilityai/sd-turbo", subfolder="unet"
        )

        if pretrained_path is not None:
            print(f"Loading pretrained model from {pretrained_path}")
            sd = torch.load(pretrained_path, map_location="cpu")
            vae_lora_config = LoraConfig(
                r=sd["rank_vae"],
                init_lora_weights="gaussian",
                target_modules=sd["vae_lora_target_modules"],
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name is not None:
            print(f"Loading pretrained model from {pretrained_name}")

            # Load UNet from the pretrained model
            pretrained_unet = UNet2DConditionModel.from_pretrained(pretrained_name, subfolder="unet", trust_remote_code=True)
            unet.load_state_dict(pretrained_unet.state_dict())

            # get the target modules for the vae
            target_modules_vae = [
                "conv1",
                "conv2",
                "conv_in",
                "conv_shortcut",
                "conv",
                "conv_out",
                "skip_conv_1",
                "skip_conv_2",
                "skip_conv_3",
                "skip_conv_4",
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
            ]

            target_modules = []
            for id, (name, param) in enumerate(vae.named_modules()):
                if "decoder" in name and any(
                    name.endswith(x) for x in target_modules_vae
                ):
                    target_modules.append(name)
            target_modules_vae = target_modules

            # note: freeze the vae encoder
            vae.encoder.requires_grad_(False)

            # note: load the lora adapter for vae
            vae_lora_config = LoraConfig(
                r=lora_rank_vae,
                init_lora_weights="gaussian",
                target_modules=target_modules_vae,
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

            from lora_vae import AutoencoderKL as PretrainedAutoencoderKL
            pretrained_vae = PretrainedAutoencoderKL.from_pretrained(pretrained_name, subfolder="vae", trust_remote_code=True)

            # load the pretrained vae
            vae.load_state_dict(pretrained_vae.state_dict())

            # breakpoint()

            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae

        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            target_modules_vae = []

            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            target_modules_vae = [
                "conv1",
                "conv2",
                "conv_in",
                "conv_shortcut",
                "conv",
                "conv_out",
                "skip_conv_1",
                "skip_conv_2",
                "skip_conv_3",
                "skip_conv_4",
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
            ]

            target_modules = []
            for id, (name, param) in enumerate(vae.named_modules()):
                if "decoder" in name and any(
                    name.endswith(x) for x in target_modules_vae
                ):
                    target_modules.append(name)
            target_modules_vae = target_modules

            # note: freeze the vae encoder
            vae.encoder.requires_grad_(False)

            # note: load the lora adapter for vae
            vae_lora_config = LoraConfig(
                r=lora_rank_vae,
                init_lora_weights="gaussian",
                target_modules=target_modules_vae,
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae

        # unet.enable_xformers_memory_efficient_attention()
        unet.to("cuda")
        vae.to("cuda")

        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1

        # note: set the sample timestep for one step denoising
        self.timesteps = torch.tensor([timestep], device="cuda").long()

        # note: freeze the text encoder
        self.text_encoder.requires_grad_(False)

        # print number of trainable parameters
        print("=" * 50)
        print(
            f"Number of trainable parameters in UNet: {sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e6:.2f}M"
        )
        print(
            f"Number of trainable parameters in VAE: {sum(p.numel() for p in vae.parameters() if p.requires_grad) / 1e6:.2f}M"
        )
        print("=" * 50)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.unet.requires_grad_(True)

        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def forward(self, x, timesteps=None, prompt=None, prompt_tokens=None):
        # either the prompt or the prompt_tokens should be provided
        assert (prompt is None) != (
            prompt_tokens is None
        ), "Either prompt or prompt_tokens should be provided"
        assert (timesteps is None) != (
            self.timesteps is None
        ), "Either timesteps or self.timesteps should be provided"

        # note: encode the text prompt and get the text embeddings
        if prompt is not None:
            # encode the text prompt
            caption_tokens = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        num_views = x.shape[1]
        x = rearrange(x, "b v c h w -> (b v) c h w")

        # note: encode the input views and get the latent embeddings
        z = self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor
        caption_enc = repeat(caption_enc, "b n c -> (b v) n c", v=num_views)

        unet_input = z

        # note: forward pass through the unet
        model_pred = self.unet(
            unet_input,
            self.timesteps,
            encoder_hidden_states=caption_enc,
        ).sample
        # note: denoise the latent with the scheduler
        z_denoised = self.sched.step(
            model_pred, self.timesteps, z, return_dict=True
        ).prev_sample
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks

        # note: decode the latent to get the output image
        output_image = (
            self.vae.decode(z_denoised / self.vae.config.scaling_factor).sample
        ).clamp(-1, 1)
        output_image = rearrange(output_image, "(b v) c h w -> b v c h w", v=num_views)

        return output_image

    def sample(
        self,
        image,
        width,
        height,
        ref_image=None,
        timesteps=None,
        prompt=None,
        prompt_tokens=None,
    ):
        input_width, input_height = image.size
        new_width = image.width - image.width % 8
        new_height = image.height - image.height % 8
        image = image.resize((new_width, new_height), Image.LANCZOS)

        T = transforms.Compose(
            [
                transforms.Resize((height, width), interpolation=Image.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        if ref_image is None:
            x = T(image).unsqueeze(0).unsqueeze(0).cuda()
        else:
            ref_image = ref_image.resize((new_width, new_height), Image.LANCZOS)
            x = torch.stack([T(image), T(ref_image)], dim=0).unsqueeze(0).cuda()

        output_image = self.forward(x, timesteps, prompt, prompt_tokens)[:, 0]
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize((input_width, input_height), Image.LANCZOS)

        return output_pil

    def save_model(self, outf, optimizer):
        sd = {}
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = self.unet.state_dict()
        sd["state_dict_vae"] = {
            k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k
        }

        sd["optimizer"] = optimizer.state_dict()

        torch.save(sd, outf)
