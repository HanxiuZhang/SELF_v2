import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # set to hf-mirror if you cannot access huggingface.co
os.environ['HF_HOME'] = '/mnt/ssd1/zhanghanxiu/cache/huggingface'
os.environ["CUDA_VISIBLE_DEVICES"] = '2' # set to your GPU id

import torch
torch.set_num_threads(4)
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../fingerprint/')))
from utils import *
aug_save_path = "aug_data"

def seed_everything(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"Global seed set to {seed}")

def add_noise(tensor, epsilon):
    noise = torch.randn_like(tensor) * epsilon
    return tensor + noise
def add_noise_qkvo(q_list, k_list, v_list, o_list, epsilon=0.1):
    new_q_list = []
    new_k_list = []
    new_v_list = []
    new_o_list = []   
    for q, k, v, o in zip(q_list, k_list, v_list, o_list):
        q_noisy = add_noise(q, epsilon)
        k_noisy = add_noise(k, epsilon)
        v_noisy = add_noise(v, epsilon)
        o_noisy = add_noise(o, epsilon)
        new_q_list.append(q_noisy)
        new_k_list.append(k_noisy)
        new_v_list.append(v_noisy)
        new_o_list.append(o_noisy)
    return new_q_list, new_k_list, new_v_list, new_o_list

def del_rows(tensor, del_num):
    indices = torch.randperm(tensor.shape[0])[:tensor.shape[0]-del_num]
    return torch.index_select(tensor, 0, indices)

def del_cols(tensor, del_num):
    indices = torch.randperm(tensor.shape[1])[:tensor.shape[1]-del_num]
    return torch.index_select(tensor, 1, indices)

def del_rows_qkvo(q_list, k_list, v_list, o_list, del_num=10, head_dim=128):
    new_q_list = []
    new_k_list = []
    new_v_list = []
    new_o_list = []
    for q, k, v, o in zip(q_list, k_list, v_list, o_list):
        if q.shape[0] != k.shape[0]:
            k = extend_kv(k, kv_head_num=k.shape[0]//head_dim, q_head_num=q.shape[0]//head_dim, head_dim=head_dim)
            v = extend_kv(v, kv_head_num=v.shape[0]//head_dim, q_head_num=q.shape[0]//head_dim, head_dim=head_dim)
        q_del = del_rows(q, del_num)
        k_del = del_rows(k, del_num)
        v_del = del_rows(v, del_num)
        o_del = del_cols(o, del_num)
        new_q_list.append(q_del)
        new_k_list.append(k_del)
        new_v_list.append(v_del)
        new_o_list.append(o_del)
    return new_q_list, new_k_list, new_v_list, new_o_list

def del_cols_qkvo(q_list, k_list, v_list, o_list, del_num=10, head_dim=128):
    new_q_list = []
    new_k_list = []
    new_v_list = []
    new_o_list = []
    for q, k, v, o in zip(q_list, k_list, v_list, o_list):
        if q.shape[0] != k.shape[0]:
            k = extend_kv(k, kv_head_num=k.shape[0]//head_dim, q_head_num=q.shape[0]//head_dim, head_dim=head_dim)
            v = extend_kv(v, kv_head_num=v.shape[0]//head_dim, q_head_num=q.shape[0]//head_dim, head_dim=head_dim)
        q_del = del_cols(q, del_num)
        k_del = del_cols(k, del_num)
        v_del = del_cols(v, del_num)
        o_del = del_rows(o, del_num)
        new_q_list.append(q_del)
        new_k_list.append(k_del)
        new_v_list.append(v_del)
        new_o_list.append(o_del)
    return new_q_list, new_k_list, new_v_list, new_o_list

def mask_tensor(tensor, ratio):
    noise = torch.rand_like(tensor)
    threshold = ratio
    mask = (noise > threshold).float()
    return tensor * mask
def mask_qkvo(q_list, k_list, v_list, o_list, ratio=0.1):
    new_q_list = []
    new_k_list = []
    new_v_list = []
    new_o_list = []
    for q, k, v, o in zip(q_list, k_list, v_list, o_list):
        q_masked = mask_tensor(q, ratio)
        k_masked = mask_tensor(k, ratio)
        v_masked = mask_tensor(v, ratio)
        o_masked = mask_tensor(o, ratio)
        new_q_list.append(q_masked)
        new_k_list.append(k_masked)
        new_v_list.append(v_masked)
        new_o_list.append(o_masked)
    return new_q_list, new_k_list, new_v_list, new_o_list

def main():
    seed_everything(42)
    select_index = [0,1,2,3,4,5,6,7]

    base_model_names = ["Qwen/Qwen2.5-7B","meta-llama/Llama-2-7b-hf","baffo32/decapoda-research-llama-7B-hf"]
    unrelated_model_list = ["mosaicml/mpt-7b","mistralai/Mistral-7B-v0.3","internlm/internlm2_5-7b"]
    related_model_list = ["meta-llama/Llama-2-7b-chat-hf"]

    for model_name in tqdm(base_model_names + unrelated_model_list + related_model_list, desc="Processing Models"):
        # Extract q, k from base_model_name_1
        q_list, k_list = extract_q_k(model_name)
        v_list, o_list = extract_v_o(model_name)
        
        # Augmentations
        qk_augmented = {}
        vo_augmented = {}

        # Add noise with different epsilons
        for epsilon in tqdm([0.1, 1, 10], desc=f"{model_name} - Adding Noise"):
            new_q_list, new_k_list, new_v_list, new_o_list = add_noise_qkvo(q_list, k_list, v_list, o_list, epsilon=epsilon)
            qk_augmented[f"add_noise_eps_{epsilon}"] = (new_q_list, new_k_list)
            vo_augmented[f"add_noise_eps_{epsilon}"] = (new_v_list, new_o_list)
            
        head_dim = get_head_dim_from_model_name(model_name)

        # Delete rows with different del_num
        for del_num in tqdm([10, 100, 1000], desc=f"{model_name} - Deleting Rows"):
            new_q_list, new_k_list, new_v_list, new_o_list = del_rows_qkvo(q_list, k_list, v_list, o_list, del_num=del_num, head_dim=head_dim)
            qk_augmented[f"del_rows_{del_num}"] = (new_q_list, new_k_list)
            vo_augmented[f"del_rows_{del_num}"] = (new_v_list, new_o_list)

        # Delete cols with different del_num
        for del_num in tqdm([10, 100, 1000], desc=f"{model_name} - Deleting Columns"):
            new_q_list, new_k_list, new_v_list, new_o_list = del_cols_qkvo(q_list, k_list, v_list, o_list, del_num=del_num, head_dim=head_dim)
            qk_augmented[f"del_cols_{del_num}"] = (new_q_list, new_k_list)
            vo_augmented[f"del_cols_{del_num}"] = (new_v_list, new_o_list)
        # Mask with different ratios
        for ratio in tqdm([0.1, 0.25, 0.5], desc=f"{model_name} - Masking"):
            new_q_list, new_k_list, new_v_list, new_o_list = mask_qkvo(q_list, k_list, v_list, o_list, ratio=ratio)
            qk_augmented[f"mask_{ratio}"] = (new_q_list, new_k_list)
            vo_augmented[f"mask_{ratio}"] = (new_v_list, new_o_list)
        
        # For each augmentation, extract singular values, eigenvalues, and save
        qk_ev_filename = f"{aug_save_path}/qk_ev_{model_name.split('/')[-1]}_augmented.csv"
        qk_sv_filename = f"{aug_save_path}/qk_sv_{model_name.split('/')[-1]}_augmented.csv"
        vo_ev_filename = f"{aug_save_path}/vo_ev_{model_name.split('/')[-1]}_augmented.csv"
        vo_sv_filename = f"{aug_save_path}/vo_sv_{model_name.split('/')[-1]}_augmented.csv"

        for aug_name, (q_aug, k_aug) in tqdm(qk_augmented.items(), desc=f"{model_name} - Saving Results"):
            print(f"Processing augmentation: {aug_name}")
            qk_sv = cal_sv_from_q_k(q_aug, k_aug, select_index)
            qk_ev = cal_ev_from_q_k(q_aug, k_aug, select_index)
            save_half_fp(f"{model_name}_{aug_name}", qk_ev, qk_ev_filename)
            save_half_fp(f"{model_name}_{aug_name}", qk_sv, qk_sv_filename)

        for aug_name, (v_aug, o_aug) in tqdm(vo_augmented.items(), desc=f"{model_name} - Saving Results"):
            print(f"Processing augmentation: {aug_name}")
            vo_sv = cal_sv_from_v_o(v_aug, o_aug, select_index)
            vo_ev = cal_ev_from_v_o(v_aug, o_aug, select_index)
            save_half_fp(f"{model_name}_{aug_name}", vo_ev, vo_ev_filename)
            save_half_fp(f"{model_name}_{aug_name}", vo_sv, vo_sv_filename)


if __name__ == "__main__":
    main()

