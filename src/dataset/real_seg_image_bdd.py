import json
import os
import random
from glob import glob
from pathlib import Path
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer


class RealSegDataset(Dataset):
    def __init__(
        self,
        real_images_path,
        seg_images_path,
        img_size,
        seg_size,
        center_crop,
        img_scale=(0.75, 1.0),
        img_ratio=(0.5625, 1.0),
        mode="default",
        cond_mode="single",
    ):
        super().__init__()

        self.img_size = img_size
        self.center_crop = center_crop
        self.seg_size = seg_size
        self.img_scale = img_scale 
        self.img_ratio = img_ratio

        self.real_images_path = list(glob(f"{real_images_path}/*.jpg", recursive=True))
        self.tgt_images_path = list(glob(f"{seg_images_path}/*.png", recursive=True))
        self.num_real_images = len(self.real_images_path)
        self.seg_images_path = list(glob(f"{seg_images_path}/*.png", recursive=True))
        self.num_seg_images = len(self.seg_images_path)
        self._length = self.num_real_images

        self.tokenizer = CLIPTokenizer.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="text_encoder").to(device="cuda")

        self.mode = mode
        assert self.mode in ["default", "panoptic",  "retrieve_adj", "retrieve_dino"]
        self.cond_mode = cond_mode
        
        self.random_resized_crop = transforms.RandomCrop(
            self.img_size, 
        )
        
        self.transform = transforms.Compose(
            [
                self.random_resized_crop,
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.cond_transform = transforms.Compose(
            [
                self.random_resized_crop,
                transforms.ToTensor()
            ]
        )

        self.random_resized_crop_aug = transforms.RandomCrop(
            self.img_size, 
        )
        
        self.aug_transform = transforms.Compose(
            [
                self.random_resized_crop_aug,
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.aug_cond_transform = transforms.Compose(
            [
                self.random_resized_crop_aug,
                transforms.ToTensor()
            ]
        )
    
    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def __getitem__(self, index):
        example = {}
        seg_image_name = str(self.seg_images_path[index % self.num_seg_images])
        mode_image_name = seg_image_name

        seg_image = Image.open(mode_image_name)
        tgt_image_name = seg_image_name.replace("/labels/sem_seg/colormaps/train/", "/images/10k/train/").replace(".png", ".jpg")
        tgt_image = Image.open(tgt_image_name) 
        json_name = mode_image_name.replace('/colormaps/train/', '/customized_sem_seg_train/').rsplit('/', 1)[0] + '.json'
        example["seg_json_name"] = json_name

        if self.mode in ["default", "panoptic"]:
            ref_seg_img_idx = random.randint(0, self.num_seg_images)
            while ref_seg_img_idx == index :
                ref_seg_img_idx = random.randint(0, self.num_seg_images)

            ref_seg_image_name = str(self.seg_images_path[ref_seg_img_idx % self.num_seg_images])
            ref_seg_image = Image.open(ref_seg_image_name) 
            ref_image_name = ref_seg_image_name.replace("/labels/sem_seg/colormaps/train/", "/images/10k/train/").replace(".png", ".jpg")
        ref_image = Image.open(ref_image_name)

        
        ref_aug_seg_image = Image.open(mode_image_name)
        ref_aug_image = Image.open(tgt_image_name)

        
        if not tgt_image.mode == "RGB":
            tgt_image = tgt_image.convert("RGB")
        if not seg_image.mode == "RGB":
            seg_image = seg_image.convert("RGB")
        if not ref_image.mode == "RGB":
            ref_image = ref_image.convert("RGB")
        if not ref_seg_image.mode == "RGB":
            ref_seg_image = ref_seg_image.convert("RGB")
            
        if not ref_aug_seg_image.mode == "RGB":
            ref_aug_seg_image = ref_aug_seg_image.convert("RGB")
        if not ref_aug_image.mode == "RGB":
            ref_aug_image = ref_aug_image.convert("RGB")

        rng_state = torch.get_rng_state()
        ref_image = self.augmentation(ref_image, self.transform)
        ref_seg_image = self.augmentation(ref_seg_image, self.cond_transform)
        
        
        tgt_image = self.augmentation(tgt_image, self.transform, rng_state)
        seg_image = self.augmentation(seg_image, self.cond_transform, rng_state)
        aug_state= torch.get_rng_state()
        
        ref_aug_seg_image = self.augmentation(ref_aug_seg_image, self.aug_cond_transform, aug_state)
        ref_aug_image = self.augmentation(ref_aug_image, self.aug_transform, aug_state)
    
        if True :
            os.makedirs("dataset_vis", exist_ok=True)
            cv2.imwrite("dataset_vis/seg_image.png", seg_image.permute(1,2,0).detach().cpu().numpy()[..., ::-1] * 255.0)
            cv2.imwrite("dataset_vis/tgt_image.png", tgt_image.permute(1,2,0).detach().cpu().numpy()[..., ::-1] * 255.0)
            cv2.imwrite("dataset_vis/ref_image.png", ref_image.permute(1,2,0).detach().cpu().numpy()[..., ::-1] * 255.0)
            cv2.imwrite("dataset_vis/ref_seg_image.png", ref_seg_image.permute(1,2,0).detach().cpu().numpy()[..., ::-1] * 255.0)
            cv2.imwrite("dataset_vis/ref_aug_image.png", ref_aug_image.permute(1,2,0).detach().cpu().numpy()[..., ::-1] * 255.0)
            cv2.imwrite("dataset_vis/ref_aug_seg_image.png", ref_aug_seg_image.permute(1,2,0).detach().cpu().numpy()[..., ::-1] * 255.0)
        if self.cond_mode == "single":
            pass

        
        example["img"] = tgt_image 
        example["seg_image"] = seg_image
        example["ref_image"] = ref_image
        example["seg_image_name"] = seg_image_name
        example["ref_seg_image_name"] = ref_seg_image_name
        example["ref_seg_image"] = ref_seg_image

        example["ref_aug_image"] = ref_aug_image 
        example["ref_aug_seg_image"] = ref_aug_seg_image
        
        return example

    def __len__(self):
        return self._length
