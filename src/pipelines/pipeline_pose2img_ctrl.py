import inspect
import json
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (DDIMScheduler, DPMSolverMultistepScheduler,
                                  EulerAncestralDiscreteScheduler,
                                  EulerDiscreteScheduler, LMSDiscreteScheduler,
                                  PNDMScheduler)
from diffusers.utils import BaseOutput, is_accelerate_available
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from src.models.mutual_self_attention import ReferenceAttentionControl
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer


@dataclass
class Pose2ImagePipelineOutput(BaseOutput):
    images: Union[torch.Tensor, np.ndarray]


class Pose2ImagePipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae,
        image_encoder,
        reference_unet,
        denoising_unet,
        pose_guider,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder").to(device="cuda")
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.cond_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents, return_type="pt"):
        latents = latents.squeeze(2)
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)["sample"]
        if return_type == "np":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_condition(
        self,
        cond_image,
        width,
        height,
        device,
        dtype,
        do_classififer_free_guidance=False,
    ):
        image = self.cond_image_processor.preprocess(
            cond_image, height=height, width=width
        ).to(dtype=torch.float32)

        image = image.to(device=device, dtype=dtype)

        if do_classififer_free_guidance:
            image = torch.cat([image] * 2)

        return image

    @torch.no_grad()
    def __call__(
        self,
        ref_image,
        ref_cond_image,
        pose_image,
        cond_image,
        # ref_json,
        # pose_json,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1        
        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        
        num_channels_latents = self.denoising_unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            self.vae.dtype,#clip_image_embeds.dtype,
            device,
            generator,
        )
        latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image, height=height, width=width
        )  # (bs, c, width, height)
        ref_image_tensor = ref_image_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)

        # Prepare pose condition image
        pose_cond_tensor = self.cond_image_processor.preprocess(
            pose_image, height=height, width=width
        )
        pose_cond_tensor = pose_cond_tensor.to(
            device=device, dtype=self.pose_guider.dtype
        )
        
        # Prepare pose condition of reference image 
        ref_pose_cond_tensor = self.cond_image_processor.preprocess(
            ref_cond_image, height=height, width=width
        )
        ref_pose_cond_tensor = ref_pose_cond_tensor.to(
            device=device, dtype=self.pose_guider.dtype
        )
        
        if cond_image is not None:
            condition_tensor = self.cond_image_processor.preprocess(
                cond_image, height=height, width=width
            )
            condition_tensor = condition_tensor.unsqueeze(2)
            condition_tensor = condition_tensor.to(
                device=device, dtype=self.pose_guider.dtype
            )
            
            conditions_fea = torch.cat([pose_cond_tensor, condition_tensor],dim=1 )
        else:
            conditions_fea = pose_cond_tensor
            
        

        # with open(pose_json, "r") as f:
        #     pose_data = json.load(f)
        # object_names = []
        # for object in pose_data["objects"]:
        #     object_names.append(object["label"])
        # object_names = list(set(object_names))
        # instance_prompt = "A photo of driving scene with"
        # for object in object_names:
        #     instance_prompt += f" {object},"
        instance_prompt = "A photo of driving scene"

        text_inputs = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=77,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids 
        text_embeds = self.text_encoder(input_ids.to(device="cuda"))[0]
        
        # with open(ref_json,"r") as fr:
        #     ref_data = json.load(fr)
        # ref_object_names = []
        # for object in ref_data["objects"]:
        #     ref_object_names.append(object["label"])
        # ref_instance_prompt = "A photo of driving scene with"
        # for object in ref_object_names:
        #     ref_instance_prompt += f" {object},"
        
        ref_instance_prompt = "A photo of driving scene"

        ref_text_inputs = self.tokenizer(
            ref_instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        ref_input_ids = ref_text_inputs.input_ids 
        ref_text_embeds = self.text_encoder(ref_input_ids.to(device="cuda"))[0]
        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # 1. Forward reference image
                # if image_prompt_embeds.ndim == 2:
                #     image_prompt_embeds = image_prompt_embeds.unsqueeze(1)
                
                if i == 0:
                    ref_controlnet_latents = ref_image_latents 
                    ref_cond_tensor = ref_pose_cond_tensor.to(device="cuda")
                    ref_down_block_res_samples, ref_mid_block_res_sample = self.pose_guider(
                        ref_controlnet_latents,
                        torch.zeros_like(t),
                        encoder_hidden_states=ref_text_embeds,
                        controlnet_cond=ref_cond_tensor,
                        return_dict=False,
                    )
                # self.reference_unet(
                #     ref_image_latents.repeat(
                #         (2 if do_classifier_free_guidance else 1), 1, 1, 1
                #     ),
                #     torch.zeros_like(t),
                #     encoder_hidden_states=ref_text_embeds,
                #     return_dict=False,
                # )
                    self.reference_unet(
                        ref_image_latents.repeat(
                            (2 if do_classifier_free_guidance else 1), 1, 1, 1
                        ),
                        torch.zeros_like(t),
                        encoder_hidden_states=ref_text_embeds,
                        down_block_additional_residuals=[
                            sample for sample in ref_down_block_res_samples
                        ],
                        mid_block_additional_residual=ref_mid_block_res_sample,
                        return_dict=False,
                    )
                    # 2. Update reference unet feature into denosing net
                    reference_control_reader.update(reference_control_writer)

                # 3.1 expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                
                controlnet_latents = latents 
                controlnet_latents = self.scheduler.scale_model_input(controlnet_latents, t)
                down_block_res_samples, mid_block_res_sample = self.pose_guider(
                    controlnet_latents,
                    t,
                    encoder_hidden_states=text_embeds,
                    controlnet_cond=conditions_fea,
                    # conditioning_scale=9.0,
                    # guess_mode=True, 
                    return_dict=False,
                )
        
        
                noise_pred = self.denoising_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeds,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
            reference_control_reader.clear()
            reference_control_writer.clear()
        # Post-processing
        image = self.decode_latents(latents)  # (b, c, 1, h, w)

        # Convert to tensor
        # if output_type == "tensor":
        #     image = torch.from_numpy(image)

        if not return_dict:
            return image

        return Pose2ImagePipelineOutput(images=image)
