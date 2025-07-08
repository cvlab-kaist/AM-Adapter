import cv2
import os 
import math 
import numpy as np 
import torch 
from einops import rearrange 
import torchvision
from PIL import Image, ImageDraw 
from itertools import product
from torch.nn import functional as F
def show_mask_on_image(img, mask, img_size=(256, 256)):
    img = np.float32(img) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255*(1-mask)), cv2.COLORMAP_JET)
    heatmap =np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cam = cv2.resize(cam, img_size)
    # cv2.imwrite(save_path, cam)
    return cam 

def visualize_hacked_attention(src_attn, tgt_attn, src_img, tgt_img, h_start, h_end, w_start, w_end, img_size=512, is_train=False, save_dir=None, timestep=0, mode="controlnext"):
    # assert src_attn.shape == tgt_attn.shape
    if src_attn == None and mode =="controlnext": 
        # no aug (controlnext)
        H = W = int(math.sqrt(tgt_attn.shape[-1]))
        tgt_attn =  rearrange(tgt_attn.mean(0).detach().cpu(), "(hs ws) (ht wt) -> hs ws ht wt", hs=H, ws=W, ht=H, wt=W)
        orig_tgt_img = tgt_img
        if is_train is not True: #[inference]
            for h, w in product(range(h_start, h_end), range(w_start,w_end)):
                tgt_attn_q = tgt_attn[h, w, :, :] # [H, W]
                # attn_v = tgt_attn_q.float().softmax(-1).unsqueeze(-1)
                attn_v = tgt_attn_q.unsqueeze(-1) #softmax(-1)
                attn_v = (attn_v - attn_v.min()) / (attn_v.max() - attn_v.min())
                # tgt_img should be the segmentation map 
                if isinstance(tgt_img, np.ndarray):
                    tgt_img = Image.fromarray(tgt_img).resize((img_size, img_size))
                else:
                    tgt_img = tgt_img.resize((img_size, img_size))

                draw = ImageDraw.Draw(tgt_img)
                point = (w * img_size / W, h * img_size / H)
                radius=4
                draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill = (255, 255, 0))
                tgt_img.show() 

                concatenated_img = Image.new('RGB', (img_size * 2, img_size))
                concatenated_img.paste(tgt_img, (0,0))
                attn_img = np.ones((H, W, 3), dtype=np.uint8)
                
                attn_img = show_mask_on_image(attn_img, attn_v.detach().cpu().numpy(), img_size=(img_size, img_size))
                attn_img_pil = Image.fromarray(attn_img)

                concatenated_img.paste(attn_img_pil, (img_size, 0))
                save_path = f"{save_dir}/timestep{timestep}/attn_size{(H, W)}"
                os.makedirs(f"{save_path}", exist_ok=True)
                concatenated_img.save(f"{save_path}/({h, w}).png")
                tgt_img = orig_tgt_img
    elif mode == "masactrl":
        H = W = int(math.sqrt(tgt_attn.shape[-1]))
        tgt_attn = rearrange(tgt_attn.mean(0).detach().cpu(), "(hs ws) (ht wt) -> hs ws ht wt", hs=H, ws=W, ht=H, wt=W)
        orig_tgt_img = tgt_img 
        if is_train is not True: #[inference]
            for h, w in product(range(h_start, h_end), range(w_start, w_end)):
                tgt_attn_q = tgt_attn[h, w, : ,:] # [H, W]
                # masactrl key,value from the source image
                attn_v = tgt_attn_q.unsqueeze(-1)
                attn_v = (attn_v - attn_v.min()) / (attn_v.max() - attn_v.min())

                if isinstance(tgt_img, np.ndarray):
                    tgt_img = Image.fromarray(tgt_img).resize((img_size, img_size))
                else:
                    tgt_img = tgt_img.resize((img_size, img_size))
                
                draw = ImageDraw.Draw(tgt_img)
                point = (w * img_size / W, h * img_size / H)
                radius=4
                draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] +radius ), fill=(255, 255, 0))
                tgt_img.show()

                concatenated_img = Image.new('RGB', (img_size * 2, img_size))
                concatenated_img.paste(tgt_img, (0, 0))
                attn_img = np.ones((H, W, 3), dtype=np.uint8)
                attn_img = show_mask_on_image(attn_img, attn_v.detach().cpu().numpy(), img_size=(img_size, img_size))
                attn_img_pil = Image.fromarray(attn_img)
                concatenated_img.paste(attn_img_pil, (img_size , 0))
                save_path = f"{save_dir}/timestep{timestep}/attn_size{(H, W)}"
                os.makedirs(f"{save_path}", exist_ok=True)
                concatenated_img.save(f"{save_path}/({h, w}).png")
                tgt_img = orig_tgt_img 
    elif mode == "sa_aug" or mode =="ours":
        H = W = int(math.sqrt(tgt_attn.shape[-1]))
        assert src_attn.shape == tgt_attn.shape 
        src_attn = rearrange(src_attn.mean(0).detach().cpu(), "(hs ws) (ht wt) -> hs ws ht wt", hs=H, ws=W, ht=H, wt=W)
        tgt_attn = rearrange(tgt_attn.mean(0).detach().cpu(), "(hs ws) (ht wt) -> hs ws ht wt", hs=H, ws=W, ht=H, wt=W)

        orig_tgt_img = tgt_img # seg map w/ structure
        orig_src_img = src_img # reference image w/ appearance
        import torchvision.transforms as transforms
        tgt_img = tgt_img.resize((img_size,img_size))
        tgt_img = transforms.Compose([transforms.ToTensor()])(tgt_img)
        src_img = transforms.Compose([transforms.ToTensor()])(src_img)
        empty_img = torch.ones_like(tgt_img)
        np_tgt_img = (tgt_img - tgt_img.min()) / (tgt_img.max() - tgt_img.min()) * 255.0
        np_tgt_img = torch.cat([empty_img, empty_img], dim=-1)
        np_tgt_img = np_tgt_img.squeeze().permute(1,2,0).detach().cpu().numpy()
        if is_train is not True: # inference  mode
            for h, w in product(range(h_start, h_end), range(w_start, w_end)):
                src_attn_q = src_attn[h, w, :, :] # [H, W]
                tgt_attn_q = tgt_attn[h, w, :, :] # [H, W]
                src_attn_q = src_attn_q.view(1, 1, H, W)
                tgt_attn_q = tgt_attn_q.view(1, 1, H, W)
                src_attn_q = F.interpolate(src_attn_q, size=(img_size, img_size), mode="bilinear", align_corners=True)
                tgt_attn_q = F.interpolate(tgt_attn_q, size = (img_size, img_size), mode="bilinear", align_corners=True)
                # src_attn_q = (src_attn_q - src_attn_q.min()) / (src_attn_q.max() - src_attn_q.min()) #* 255.
                # tgt_attn_q = (tgt_attn_q - tgt_attn_q.min()) / (tgt_attn_q.max() - tgt_attn_q.min()) #* 255.
                # src_attn_q = F.softmax(src_attn_q, dim=-1) # * 255.0
                # tgt_attn_q = F.softmax(tgt_attn_q, dim=-1)# * 255.0
                attn_v = torch.cat([src_attn_q, tgt_attn_q], dim=-1)
                attn_v = F.softmax(attn_v, dim=-1) # [H, 2 * W]
                # attn_v = attn_v.float().unsqueeze(-1)
                attn_v = (attn_v - attn_v.min()) / (attn_v.max() - attn_v.min()) 

                attn_v = attn_v.squeeze().detach().cpu().numpy() * 255.
                heat_map= cv2.applyColorMap(attn_v.astype(np.uint8), cv2.COLORMAP_JET)
                # heat_map = np_tgt_img[...,::-1]  + heat_mask
                # heat_map = (heat_map - heat_map.min()) / (heat_map.max() - heat_map.min()) *255.0
                # attn_v = attn_v.reshape(H, 2 * W).float().unsqueeze(-1)
                # attn_v = attn_v.reshape(H, 2 * W).float().unsqueeze(-1)
                # import pdb; pdb.set_trace()
                # attn_v = attn_v.softmax(-1)
                # cv2.imwrite("heat_mask.png", heat_mask)
                # import pdb; pdb.set_trace()
                # if isinstance(src_img, np.ndarray):
                #     src_img = Image.fromarray(src_img).resize((img_size, img_size))
                # else:
                #     src_img = src_img.resize((img_size, img_size))
                
                # if isinstance(tgt_img, np.ndarray):
                #     tgt_img = Image.fromarray(tgt_img).resize((img_size, img_size))
                # else:
                #     tgt_img = tgt_img.resize((img_size, img_size))
                
                # draw = ImageDraw.Draw(tgt_img)
                # point = (w * img_size / W, h * img_size / H)
                # radius=4
                # draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] +radius ), fill=(255, 255, 0))
                # tgt_img.show()

                # concatenated_img = Image.new('RGB', (img_size * 4, img_size))
                # concatenated_img.paste(src_img, (0, 0))
                # concatenated_img.paste(tgt_img, (img_size, 0))
                # attn_img = np.ones((H, W * 2, 3), dtype=np.uint8)
                # attn_img = show_mask_on_image(attn_img, attn_v.detach().cpu().numpy(), img_size=(img_size * 2, img_size))
                # attn_img_pil = Image.fromarray(attn_img)
                # concatenated_img.paste(attn_img_pil, (img_size * 2, 0))
                save_path = f"{save_dir}/timestep{timestep}/attn_size{(H, W)}"
                os.makedirs(f"{save_path}", exist_ok=True)
                cv2.imwrite(f"{save_path}/({h, w}).png", heat_map)
                # concatenated_img.save(f"{save_path}/({h, w}).png")
                tgt_img = orig_tgt_img
                src_img = orig_src_img
                