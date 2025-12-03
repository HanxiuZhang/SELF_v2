import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # set to hf-mirror if you cannot access huggingface.co
os.environ['HF_HOME'] = '/mnt/ssd1/zhanghanxiu/cache/huggingface'
os.environ["CUDA_VISIBLE_DEVICES"] = '2' # set to your GPU id

import torch
torch.set_num_threads(16)
import os
from simnet import SimilarityNet

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../fingerprint/')))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
import random
from datetime import datetime
aug_path = 'augmentation/aug_data'
simnet_path = '/mnt/ssd1/zhanghanxiu/model/self_simnet'
from utils import *
import time, atexit

qk_ev_filename = "../fingerprint/fp_data/qk_ev.csv" # filename to save eigenvalues
qk_sv_filename = "../fingerprint/fp_data/qk_sv.csv" # filename to save singular values
vo_ev_filename = "../fingerprint/fp_data/vo_ev.csv" # filename to save eigenvalues
vo_sv_filename = "../fingerprint/fp_data/vo_sv.csv" # filename to save singular values

file_list = [qk_ev_filename, qk_sv_filename, vo_ev_filename, vo_sv_filename]

def seed_everything(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"Global seed set to {seed}")

def construct_dataset(base_model_name, related_list=[], unrelated_list=[]):
    data = []

    positive_label = 1
    negative_label = 0

    for model_name in [base_model_name]+related_list+unrelated_list:
        print(f"Add model: {model_name}")
        fp = get_fp_new(model_name,file_list=file_list).unsqueeze(0).to(device).float()
        label = positive_label if model_name in [base_model_name]+related_list else negative_label
        data.append((fp, label))

        model_save_name = model_name.split("/")[-1]

        aug_qk_ev_filename = f'{aug_path}/qk_ev_{model_save_name}_augmented.csv'
        aug_qk_sv_filename = f'{aug_path}/qk_sv_{model_save_name}_augmented.csv'
        aug_vo_ev_filename = f'{aug_path}/vo_ev_{model_save_name}_augmented.csv'
        aug_vo_sv_filename = f'{aug_path}/vo_sv_{model_save_name}_augmented.csv'

        aug_file_list = [aug_qk_ev_filename, aug_qk_sv_filename, aug_vo_ev_filename, aug_vo_sv_filename]
        
        aug_methods = ['add_noise_eps', 'del_rows', 'del_cols', 'mask']
        eps_list = [0.1,1,10]
        del_num_list = [10,100,1000]
        ratio_list = [0.1,0.25,0.5]

        for aug_method in aug_methods:

            if aug_method == 'add_noise_eps':
                aug_list = eps_list
            elif aug_method == 'del_rows' or aug_method == 'del_cols':
                aug_list = del_num_list
            else:
                aug_list = ratio_list

            for aug in aug_list:
                fp_aug = get_fp_new(f'{model_name}_{aug_method}_{aug}', file_list=aug_file_list).unsqueeze(0).to(device).float()

                data.append((fp_aug, label))

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--step_gamma", type=float, default=0.8)
    parser.add_argument("--adv_eps", type=float, default=1e-5)
    parser.add_argument("--smooth", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--tag", type=str, default=timestamp)
    args = parser.parse_args()

    seed_everything(42)

    base_model_name = "meta-llama/Llama-2-7b-hf"
    # base_model_name = "baffo32/decapoda-research-llama-7B-hf"
    related_list = ["meta-llama/Llama-2-7b-chat-hf"]
    unrelated_list = ["mistralai/Mistral-7B-v0.3","mosaicml/mpt-7b","internlm/internlm2_5-7b"]

    # data = construct_dataset(base_model_name, related_list, unrelated_list)
    data = construct_dataset(base_model_name=base_model_name, related_list=related_list, unrelated_list=unrelated_list)
    inputs = torch.cat([item[0] for item in data], dim=0)
    labels = torch.tensor([item[1] for item in data], dtype=torch.float32).to(device)
    dataset = TensorDataset(inputs,labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = SimilarityNet().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    # Adversarial training parameters
    epsilon = args.adv_eps

    smoothing = args.smooth
    num_epochs = args.epochs
    print(num_epochs)
    losses = []
    train_start_time = time.time()
    train_start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Training started at {train_start_dt}")

    def _print_training_time():
        elapsed = time.time() - train_start_time
        hrs, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        print(f"\nTotal training time: {int(hrs)}h {int(mins)}m {secs:.2f}s")

    atexit.register(_print_training_time)
    
    for epoch in range(num_epochs):
        for vec_batch, label_batch in dataloader:
            vec_batch = vec_batch.detach().clone().requires_grad_(True)

            # Label smoothing
            label_batch = label_batch * (1 - smoothing) + 0.5 * smoothing

            # Forward pass
            outputs = model(vec_batch)
            loss = criterion(outputs, label_batch)
            loss.backward(retain_graph=True)

            # Generate adversarial examples
            grad_vec = vec_batch.grad.detach()
            vec_adv = vec_batch + epsilon * torch.sign(grad_vec)

            # Clear gradients and train again
            optimizer.zero_grad()
            outputs_adv = model(vec_adv)
            loss_adv = criterion(outputs_adv, label_batch)
            loss_adv.backward()
            optimizer.step()

            scheduler.step()
            losses.append(loss_adv.item())
            print(f"\rEpoch [{epoch+1}/{num_epochs}], Loss: {loss_adv.item():.8f}", end="")

    model_save_name = base_model_name.split("/")[-1]
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{simnet_path}/simnet_{model_save_name}_{args.tag}.pth"
    torch.save(model, save_path)
    print(f"Model saved as {save_path}")

if __name__ == "__main__":
    main()

    