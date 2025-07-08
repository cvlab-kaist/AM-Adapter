import argparse
import gc
import json
import logging
import math
import os
import os.path as osp
import random
import warnings
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import diffusers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import (AutoencoderKL,  DDIMScheduler, DDPMScheduler,
                       StableDiffusionControlNetPipeline, UniPCMultistepScheduler)
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from PIL import Image
from src.dataset.real_seg_image_bdd import RealSegDataset
from src.models.unet_2d_condition import UNet2DConditionModel

from src.utils.util import (delete_additional_ckpt, import_filename,
                            seed_everything)
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                          CLIPTokenizer, CLIPVisionModelWithProjection)
from src.models.controlnext import ControlNeXtModel
from safetensors.torch import load_file
from diffusers.models.attention_processor import XFormersAttnProcessor, LoRAXFormersAttnProcessor

from src.pipelines.pipeline_ours import StableDiffusionControlNeXtPipeline as Seg2ImagePipeline
from src.models.matching_module import OurModel
from src.models.attention import BasicTransformerBlock
from src.models.attention_processor import MatchAttnProcessor

def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result
    
def set_up_match_attn_processor_(blocks, block_type="down"):
    blocks_ = [m for m in torch_dfs(blocks) if isinstance(m,BasicTransformerBlock)]
    for idx, m in enumerate(blocks_):
        embed_dim = 32 
        processor = MatchAttnProcessor(
            embed_dim=embed_dim,
            hidden_size=m.attn1.to_q.out_features,
        )
        processor.requires_grad_(True)
        m.attn1.processor = processor 

def set_up_match_attn_processor(unet, fusion_blocks):
    device, dtype = unet.conv_in.weight.device, unet.conv_in.weight.dtype
    scale_idx=0
    
    if fusion_blocks == "full":
        set_up_match_attn_processor_(unet.down_blocks, block_type="down")
        set_up_match_attn_processor_(unet.mid_block, block_type="mid")
        set_up_match_attn_processor_(unet.up_blocks, block_type="up")
    elif fusion_blocks == "midup":
        set_up_match_attn_processor_(unet.mid_block, block_type="mid")
        set_up_match_attn_processor_(unet.up_blocks, block_type="up")
    elif fusion_blocks == "down":
        set_up_match_attn_processor_(unet.down_blocks, block_type="down")
    elif fusion_blocks == "up":
        set_up_match_attn_processor_(unet.up_blocks, block_type="up")
    
        
def log_validation(
    cfg,
    vae,
    appearance_unet,
    semantic_unet,
    controlnext,
    scheduler,
    matcher,
    width,
    height,
):
    generator = torch.Generator().manual_seed(42)
    seed_everything(42)
    vae = vae.to(dtype=torch.float32)
    
    pipe = Seg2ImagePipeline(
        vae=vae,
        appearance_unet=appearance_unet,
        semantic_unet=semantic_unet,
        controlnext=controlnext,
        scheduler=scheduler,
        matcher=matcher,
    )

    pipe= pipe.to("cuda", dtype=torch.float32) 
    app_image_paths = cfg.validation.real_paths 
    tgt_image_paths = cfg.validation.seg_paths
    pil_images = []
    set_up_match_attn_processor(semantic_unet, fusion_blocks="full")

    if cfg.validation.version == "retrieval":
        with open(cfg.validation.retrieval_json, "r") as f:
            data = json.load(f)
            tgt_image_paths = list(data.keys()) 
            for tgt_name in tqdm(tgt_image_paths):
                app_name_list = data[tgt_name]
                app_name = app_name_list[0]
                root_path = cfg.validation.validation_root_path 
                app_seg_path = os.path.join(cfg.validation.validation_seg_root_path, f"{app_name}.png")
                tgt_seg_path = os.path.join(cfg.validation.validation_seg_root_path, f"{tgt_name}.png")
                if os.path.exists(app_seg_path) and os.path.exists(tgt_seg_path):
                    app_seg_pil = Image.open(app_seg_path).convert("RGB")
                    tgt_seg_pil = Image.open(tgt_seg_path).convert("RGB")
                    app_image_path = os.path.join(cfg.validation.validation_real_root_path, f"{app_name}.jpg")
                    app_image_pil = Image.open(app_image_path).convert("RGB")
                    save_name = None 
                    image = pipe(
                        prompt="A photo of a <sks> driving scene",
                        app_image=app_image_pil, 
                        app_seg_image=app_seg_pil, 
                        tgt_seg_image=tgt_seg_pil,
                        height=height, 
                        width=width, 
                        num_inference_steps=20, 
                        guidance_scale=7.5,
                        generator=generator,
                        save_name=save_name,
                        mode="ours",
                    ).images

                    res_image_pil = image[0]
                    w, h = res_image_pil.size
                   
                    if cfg.validation.save_type == "triplet" : # app, tgt, result 
                        canvas = Image.new("RGB", (w * 3, h), "white")
                        app_image_pil = app_image_pil.resize((w, h))
                        tgt_seg_pil = tgt_seg_pil.resize((w, h))
                        canvas.paste(app_image_pil, (0, 0))
                        canvas.paste(tgt_seg_pil, (w, 0))
                        canvas.paste(res_image_pil, (w * 2, 0))
                    else: 
                        canvas = Image.new("RGB", (w, h), "white")
                        canvas.paste(res_image_pil, (0,0))
                    pil_images.append({"name": f"{app_name}_{tgt_name}", "img": canvas})
                    os.makedirs(f"{cfg.output_dir}/", exist_ok=True)
                    sample_name = f"{app_name}_{tgt_name}"
                    img = canvas 
                    with TemporaryDirectory() as temp_dir:
                        os.makedirs(f"{temp_dir}", exist_ok=True)
                        out_file =f"{cfg.output_dir}/{sample_name}.png"

                        img.save(out_file)           
    else:
        for app_image_path in app_image_paths:
            for tgt_image_path in tgt_image_paths:
                tgt_name = tgt_image_path.split("/")[-1]
                app_name = app_image_path.split("/")[-1]
                app_image_pil = Image.open(app_image_path).convert("RGB")
                tgt_image_pil = Image.open(tgt_image_path).convert("RGB")
                app_seg_path = os.path.join(cfg.validation.validation_seg_root_path, f"{app_name}.png")
                app_seg_pil = Image.open(app_seg_path).convert("RGB")
                save_name = None
                image = pipe(
                    prompt="A photo of a <sks> driving scene",
                    ref_image=app_image_pil,
                    ref_seg_image=app_seg_pil,
                    tgt_seg_image=tgt_image_pil,
                    height=height,
                    width=width,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    generator=generator,
                    save_name=save_name,
                    mode="ours",
                ).images
        
                res_image_pil = image[0]
                w, h = res_image_pil.size
                
                if cfg.validation.save_type == "triplet" : # app, tgt, result 
                    canvas = Image.new("RGB", (w * 3, h), "white")
                    app_image_pil = app_image_pil.resize((w, h))
                    tgt_seg_pil = tgt_seg_pil.resize((w, h))
                    canvas.paste(app_image_pil, (0, 0))
                    canvas.paste(tgt_seg_pil, (w, 0))
                    canvas.paste(res_image_pil, (w * 2, 0))
                else: 
                    canvas = Image.new("RGB", (w, h), "white")
                    canvas.paste(res_image_pil, (0,0))
                pil_images.append({"name": f"{app_name}_{tgt_name}", "img": canvas})
                os.makedirs(f"{cfg.output_dir}/", exist_ok=True)
                sample_name = f"{app_name}_{tgt_name}"
                img = canvas 
                with TemporaryDirectory() as temp_dir:
                    os.makedirs(f"{temp_dir}", exist_ok=True)
                    out_file =f"{cfg.output_dir}/{sample_name}.png"

                    img.save(out_file)  
         
    vae = vae.to(dtype=torch.float16)

    del vae
    del pipe
    torch.cuda.empty_cache()

    return pil_images

def load_safetensors(model, safetensors_path, strict=True, load_weight_increasement=False):
    if not load_weight_increasement:
        if safetensors_path.endswith('.safetensors'):
            state_dict = load_file(safetensors_path)
        else:
            state_dict = torch.load(safetensors_path)
        model.load_state_dict(state_dict, strict=strict)
    else:
        if safetensors_path.endswith('.safetensors'):
            state_dict = load_file(safetensors_path)
        else:
            state_dict = torch.load(safetensors_path)
        pretrained_state_dict = model.state_dict()
        for k in state_dict.keys():
            state_dict[k] = state_dict[k] + pretrained_state_dict[k]
        model.load_state_dict(state_dict, strict=False)

def load_models(cfg):
    appearance_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    )
    semantic_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    )
    weight_dtype=torch.float32
    controlnext = ControlNeXtModel(controlnext_scale=1.0)
    if cfg.pretrained_unet_path is not None:
        load_safetensors(appearance_unet, cfg.pretrained_unet_path, strict=True, load_weight_increasement=False)
        load_safetensors(semantic_unet, cfg.pretrained_unet_path, strict=True, load_weight_increasement=False)
    if cfg.net.controlnext_path is not None:
        load_safetensors(controlnext, cfg.net.controlnext_path, strict=True)
    
    appearance_unet.enable_xformers_memory_efficient_attention()
    semantic_unet.enable_xformers_memory_efficient_attention()
    semantic_unet.to(device="cuda", dtype=weight_dtype) 
    
    if cfg.net.matcher_path is not None:
        set_up_match_attn_processor(semantic_unet, fusion_blocks="full")
        matcher = OurModel()
        matcher_ = torch.load(cfg.net.matcher_path)
        matcher.load_state_dict(matcher_)
 
    appearance_unet.to(device="cuda",dtype=weight_dtype)
    controlnext.to(dtype=weight_dtype)
    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)

    val_noise_scheduler_config = {'num_train_timesteps': 1000, 
                        'beta_start': 0.00085, 
                        'beta_end': 0.012, 
                        'beta_schedule': 'scaled_linear', 
                        'trained_betas': None, 
                        'solver_order': 2, 
                        'prediction_type': 'epsilon', 
                        'thresholding': False,
                        'dynamic_thresholding_ratio': 0.995, 
                        'sample_max_value': 1.0, 
                        'predict_x0': True, 
                        'solver_type': 'bh2', 
                        'lower_order_final': True, 
                        'disable_corrector': [], 
                        'solver_p': None, 
                        'use_karras_sigmas': False, 
                        'use_exponential_sigmas': False, 
                        'use_beta_sigmas': False, 
                        'timestep_spacing': 'linspace', 
                        'steps_offset': 1, 
                        'final_sigmas_type': 'zero', 
                        '_use_default_values': ['prediction_type', 'disable_corrector', 'use_beta_sigmas', 'solver_p', 'sample_max_value', 'use_exponential_sigmas', 'use_karras_sigmas', 'predict_x0', 'timestep_spacing', 'lower_order_final', 'thresholding', 'rescale_betas_zero_snr', 'dynamic_thresholding_ratio', 'solver_order', 'solver_type', 'final_sigmas_type'],
                        'skip_prk_steps': True, 
                        'set_alpha_to_one': False, 
                        '_class_name': 'UniPCMultistepScheduler', 
                        '_diffusers_version': '0.6.0', 
                        'clip_sample': False}
    val_noise_scheduler = DDIMScheduler.from_config(val_noise_scheduler_config)

    vae = AutoencoderKL.from_pretrained(
        cfg.vae_model_path, subfolder="vae"
    ).to("cuda", dtype=torch.float16)

    
    return appearance_unet, semantic_unet, controlnext, val_noise_scheduler, matcher, vae

def main(cfg):
    appearance_unet, semantic_unet, controlnext, scheduler, matcher, vae = load_models(config)  
    appearance_unet.eval()
    semantic_unet.eval() 
    controlnext.eval()
    vae.eval()
    matcher.eval()
    sample_dicts = log_validation(
        cfg=cfg,
        vae=vae, 
        appearance_unet=appearance_unet,
        semantic_unet=semantic_unet,
        controlnext=controlnext,
        scheduler=scheduler,
        matcher=matcher,
        width=cfg.data.val_width,
        height=cfg.data.val_height,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--config", type=str, default="./configs/inference_adapter.yaml")
    args = parser.parse_args() 
    
    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    
    main(config)
    