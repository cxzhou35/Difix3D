import json
import os

import torch
import torchvision.transforms.functional as F
from PIL import Image


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, dataset_path, split, height=576, width=1024, tokenizer=None):

        super().__init__()
        with open(dataset_path, "r") as f:
            self.data = json.load(f)[split]  # split: train/test
        self.root = root_dir
        self.img_ids = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]

        input_img = self.data[img_id]["image"]
        input_img_path = os.path.join(self.root, input_img)
        output_img = self.data[img_id]["target_image"]
        output_img_path = os.path.join(self.root, output_img)
        ref_img = (
            self.data[img_id]["ref_image"] if "ref_image" in self.data[img_id] else None
        )
        ref_img_path = os.path.join(self.root, ref_img) if ref_img is not None else None
        caption = self.data[img_id]["prompt"]

        try:
            input_img = Image.open(input_img_path)
            output_img = Image.open(output_img_path)
        except:
            print("Error loading image:", input_img, output_img)
            return self.__getitem__(idx + 1)

        img_t = F.to_tensor(input_img)
        img_t = F.resize(img_t, self.image_size)
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

        output_t = F.to_tensor(output_img)
        output_t = F.resize(output_t, self.image_size)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        if ref_img is not None:
            ref_img = Image.open(ref_img_path)
            ref_t = F.to_tensor(ref_img)
            ref_t = F.resize(ref_t, self.image_size)
            ref_t = F.normalize(ref_t, mean=[0.5], std=[0.5])

            img_t = torch.stack([img_t, ref_t], dim=0)
            output_t = torch.stack([output_t, ref_t], dim=0)
        else:
            img_t = img_t.unsqueeze(0)
            output_t = output_t.unsqueeze(0)

        out = {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
        }

        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids
            out["input_ids"] = input_ids

        return out
