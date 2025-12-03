import os
 
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # set to hf-mirror if you cannot access huggingface.co 
os.environ['HF_HOME'] = '/mnt/ssd1/zhanghanxiu/cache/huggingface'
os.environ["CUDA_VISIBLE_DEVICES"] = '2' # set to your GPU id 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../fingerprint/')))
import warnings 
warnings.filterwarnings("ignore") 
import torch 
torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils import *
import argparse
import time

qk_ev_filename = "../fingerprint/fp_data/qk_ev.csv" # filename to save eigenvalues
qk_sv_filename = "../fingerprint/fp_data/qk_sv.csv" # filename to save singular values
vo_ev_filename = "../fingerprint/fp_data/vo_ev.csv" # filename to save eigenvalues
vo_sv_filename = "../fingerprint/fp_data/vo_sv.csv" # filename to save singular values

file_list = [qk_ev_filename, qk_sv_filename, vo_ev_filename, vo_sv_filename]


def main():

    parser = argparse.ArgumentParser()
    # model_dir = '/mnt/ssd1/zhanghanxiu/model/self_simnet'
    # model_dir = 'model'
    pths = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pth")]
    # default_pth = max(pths, key=os.path.getmtime) if pths else ""
    default_pth = 'model/simnet_llama2.pth'
    parser.add_argument("--simnet_path", type=str, default=default_pth, help=f"path to simnet .pth (defaults to latest in {model_dir}/)")
    args = parser.parse_args()

    simnet_path = args.simnet_path

    print(f"Load simnet from {simnet_path}")
    finetune_model_list = ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/CodeLlama-7b-hf", "lmsys/vicuna-7b-v1.5", "WizardLMTeam/WizardMath-7B-V1.0", "EleutherAI/llemma_7b", "cxllin/Llama2-7b-Finance","LinkSoul/Chinese-Llama-2-7b"]
    prune_model_list = ["princeton-nlp/Sheared-LLaMA-2.7B", "nm-testing/SparseLlama-2-7b-pruned_50.2of4", "princeton-nlp/Sheared-LLaMA-1.3B-Pruned","princeton-nlp/Sheared-LLaMA-1.3B-ShareGPT","princeton-nlp/Sheared-LLaMA-2.7B-Pruned","princeton-nlp/Sheared-LLaMA-2.7B-ShareGPT","princeton-nlp/Sheared-LLaMA-1.3B","MBZUAI-LLM/GBLM-Pruner-LLaMA-2-7B",'/mnt/ssd1/zhanghanxiu/model/prune_wanda','/mnt/ssd1/zhanghanxiu/model/prune_sparsegpt']
    unrelated_model_list = ["mistralai/Mistral-7B-v0.3", "Qwen/Qwen1.5-7B", "baichuan-inc/Baichuan2-7B-Base", "internlm/internlm2_5-7b", "openai-community/gpt2-large","cerebras/Cerebras-GPT-1.3B","zai-org/chatglm2-6b", "facebook/opt-6.7b", "EleutherAI/pythia-6.9b", "mosaicml/mpt-7b"]
    merged_model_list = ['Wanfq/FuseLLM-7B']


    print('--------------llama-2-7b unrelated--------------')
    for model_path in unrelated_model_list:
        dist = fp_new_sim_from_path(model_path, simnet_path, file_list=file_list)
        print(f"{model_path:60} {dist:>10.8f}")

    print('--------------llama-2-7b finetune--------------')
    for model_path in finetune_model_list:
        dist = fp_new_sim_from_path(model_path, simnet_path, file_list=file_list)
        print(f"{model_path:60} {dist:>10.8f}")

    print('-------------llama-2-7b pruned--------------')
    for model_path in prune_model_list:
        dist = fp_new_sim_from_path(model_path, simnet_path, file_list=file_list)
        print(f"{model_path:60} {dist:>10.8f}")

    print('--------------llama-2-7b merged--------------')
    for model_path in merged_model_list:
        dist = fp_new_sim_from_path(model_path, simnet_path, file_list=file_list)
        print(f"{model_path:60} {dist:>10.8f}")
    

if __name__ == "__main__":
    main()
    