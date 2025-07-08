import numpy as np
from tqdm import tqdm 
def visualize_attnmaps(cross_sims, prompt, save_individual_attnmaps=False, num_inference_steps=50, save_dir=None):
    cross_sims_2d = []
    
    import os
    save_dir = f'{save_dir}/attnmaps/{prompt}'
    os.makedirs(save_dir, exist_ok=True)
        
    # process attnmaps to 2d
    print("len(cross_sims): ", len(cross_sims))

    # for sim in tqdm(cross_sims, desc="Processing attnmaps", total=len(cross_sims)):
    #     cross_sim_2d = process_cross_attnmap_to_2d(sim)
    #     cross_sims_2d.append(cross_sim_2d)
    for attn_layer in tqdm(cross_sims.keys()):
        cross_sim_2d = process_cross_attnmap_to_2d(cross_sims[attn_layer])
        cross_sims_2d.append(cross_sim_2d)
        
    from collections import defaultdict
    
    cross_sims_2d_dict = defaultdict(list)
    NUM_OF_CROSS_ATTENTION_LAYERS_TO_TRACK = len(cross_sims_2d) #// num_inference_steps
    for i, sim_2d in tqdm(enumerate(cross_sims_2d), desc="Visualizing attnmaps", total=len(cross_sims_2d)):
        cross_layer_index = i % NUM_OF_CROSS_ATTENTION_LAYERS_TO_TRACK
        cross_timestep_index = i // NUM_OF_CROSS_ATTENTION_LAYERS_TO_TRACK
        
        if save_individual_attnmaps:
            save_in_colormap(sim_2d.cpu(), f"{save_dir}/cross_attnmap_layer@{cross_layer_index}_timstep@{cross_timestep_index}.png")

        cross_sims_2d_dict[cross_layer_index].append(sim_2d.cpu())
        # print('cross_layer_index:', cross_layer_index, 'sim_2d.shape:', sim_2d.shape)
    for layer_index, sim_2ds in cross_sims_2d_dict.items():
        cross_sim_2d = np.concatenate(sim_2ds, axis=1)
        save_in_colormap(cross_sim_2d, f"{save_dir}/cross_attnmap_all_timesteps_layer@{layer_index}.png", colorbar=False)
   
def process_spatial_attnmap_to_2d(sim, reverse=False):
    # sim: [HW, 2HW]
    # Split the attention map into two separate maps
    HW = sim.shape[0]
    map1 = sim[:, :HW]
    map2 = sim[:, HW:]

    # channelwise mean
    if reverse:
        map1 = map1.mean(axis=0)
        map2 = map2.mean(axis=0)
    else:
        map1 = map1.mean(axis=1)
        map2 = map2.mean(axis=1)

    # reshape to 2d
    H = int(np.sqrt(HW))
    W = H    
    map1 = map1.reshape(H, W)
    sim = map1
    map2 = map2.reshape(H, W)
    
    # concatenate two maps
    sim = np.concatenate([map1, map2], axis=1)
    
    return sim
    
    
def process_cross_attnmap_to_2d(sim):
    HW = sim.shape[0]
    
    TARGET_TOKEN_INDEX = 2 # "a <cat-toy> toy in the jungle" -> 2
    sim = sim[:, TARGET_TOKEN_INDEX]

    # H = int(np.sqrt(HW))
    # W = H    
    # sim = sim.reshape(H, W)

    return sim


def save_in_colormap(sim, save_path, colorbar=True):
    import matplotlib.pyplot as plt
    
    plt.imshow(sim, cmap='viridis')
    if colorbar:
        plt.colorbar()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved attnmap to {save_path}")