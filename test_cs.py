import torch
import random
import sys
import os
from datasets import Dataset, load_dataset, get_dataset_config_names, concatenate_datasets, interleave_datasets
from peft import LoraConfig, TaskType, PeftModel, get_peft_model
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from data_load import *
from test_fc import *
from collections import Counter
from torch.utils.data import DataLoader

torch_dtype = torch.float32
base_model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/qwen/Qwen2.5-14B',
                                              device_map="auto",
                                              torch_dtype=torch_dtype,
                                              output_hidden_states=True
                                              )

# for name, param in model.named_parameters():
#     print(f"{name}: {param.dtype}")

tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/qwen/Qwen2.5-14B', 
                                          use_fast=False, trust_remote_code=True,
                                          padding_side='left')

task = 'arc'
save_dir = "/root/autodl-tmp/data/arc_processed"
dataset = load_from_disk(f"{save_dir}")
arc_ds = {task:dataset["validation"]}

save_dir = "/root/autodl-tmp/data/glue_processed"
DATASET = ["qnli", "cola", "mnli", "qqp", "sst2"]
_, glue_test_ds = load_ds_from_disk(save_dir, DATASET)
save_dir = "/root/autodl-tmp/data/common_sense_processed"
# DATASET = ["piqa", "siqa"]
# DATASET = ["piqa", "winogrande"]
DATASET = ["piqa", "winogrande", "siqa", "boolq", "hellaswag"]
_, cs_test_ds = load_ds_from_disk(save_dir, DATASET)

task = 'gsm8k'
save_dir = "/root/autodl-tmp/data/gsm8k_processed"
dataset = load_from_disk(f"{save_dir}")
gsm_ds = {task:dataset["test"]}

test_ds = {**glue_test_ds, **cs_test_ds, **arc_ds, **gsm_ds}
# test_ds = {**gsm_ds}
# test_ds = gsm_ds

random_seed = 7
set_seed(random_seed)
sample_size = 1000
for key in test_ds.keys():
    if len(test_ds[key]) > sample_size:
        indices = random.sample(range(len(test_ds[key])), sample_size)
        test_ds[key] = test_ds[key].select(indices)

# # 检查所有数据集的标签分布
for name, dataset in test_ds.items():
    print(name)
    # print(dataset[0])
    print(dataset)
#     check_label_distribution(dataset, f"{name} 测试集")
seeds = (0, 7, 27)
seed = 7
drops = [0.2]
# for drop in drops:
# paths = [
#     "autodl-tmp/HydraLoRA/HydraLoRA/output/3-hydra_rou_mt5_kaiming_seed7_drop0.2/checkpoint-599",
#     "autodl-tmp/HydraLoRA/HydraLoRA/output/3-hydra_zero_mt5_kaiming_seed7_drop0.2/checkpoint-599",
#     "autodl-tmp/HydraLoRA/HydraLoRA/output/3-rlora_zero_mt5_kaiming_seed7_drop0.2/checkpoint-599",
#     "autodl-tmp/HydraLoRA/HydraLoRA/output/3-rlora_rou_mt5_kaiming_seed7_drop0.2/checkpoint-599",
#     "autodl-tmp/HydraLoRA/HydraLoRA/output/3-sd_zero_mt5_kaiming_seed7_drop0.2/checkpoint-599"
#         ]
# paths = [
#     "autodl-tmp/HydraLoRA/HydraLoRA/output/3-hydra_avg_mt5_kaiming_seed7_drop0.2/checkpoint-599",
#     "autodl-tmp/HydraLoRA/HydraLoRA/output/3-rlora_avg_mt5_kaiming_seed7_drop0.2/checkpoint-599",
#     "autodl-tmp/HydraLoRA/HydraLoRA/output/3-rlora_avg_mt5_unit_seed7_drop0.2/checkpoint-599",
#     "autodl-tmp/HydraLoRA/HydraLoRA/output/3-sd_avg_mt5_kaiming_seed7_drop0.2/checkpoint-599"
#         ]
paths = [
    # "autodl-tmp/HydraLoRA/HydraLoRA/output/RDdrop_3B_rdrop_alpha0.0-rlora_unit_seed7_12tasks/checkpoint-491",
    # "autodl-tmp/HydraLoRA/HydraLoRA/output/RDdrop_3B_rdrop_alpha0.01-rlora_unit_seed7_12tasks/checkpoint-491",
    # "autodl-tmp/HydraLoRA/HydraLoRA/output/RDdrop_3B_rdrop_alpha0.02-rlora_unit_seed7_12tasks/checkpoint-491",
    # "autodl-tmp/HydraLoRA/HydraLoRA/output/RDdrop_3B_rdrop_alpha0.03-rlora_unit_seed7_12tasks/checkpoint-491",
    # "autodl-tmp/HydraLoRA/HydraLoRA/output/RDdrop_3B_rdrop_alpha0.04-rlora_unit_seed7_12tasks/checkpoint-491",
    # "autodl-tmp/HydraLoRA/HydraLoRA/output/RDdrop_3B_rdrop_alpha0.05-rlora_unit_seed7_12tasks/checkpoint-491",
    # "autodl-tmp/HydraLoRA/HydraLoRA/output/RDdrop_3B_rdrop_alpha0.06-rlora_unit_seed7_12tasks/checkpoint-491",
    # "autodl-tmp/HydraLoRA/HydraLoRA/output/RDdrop_3B_rdrop_alpha0.07-rlora_unit_seed7_12tasks/checkpoint-491",
    # "autodl-tmp/HydraLoRA/HydraLoRA/output/RDdrop_3B_rdrop_alpha0.08-rlora_unit_seed7_12tasks/checkpoint-491",
    "autodl-tmp/HydraLoRA/HydraLoRA/output/rlora_14B_gauss0.0_7B-rlora_unit_seed7_12tasks/checkpoint-491",
        ]
for path in paths: 
    # lora_path = f'/root/autodl-tmp/HydraLoRA/HydraLoRA/output/3-sd_rou_mt5_kaiming_seed7_drop0.2/checkpoint-599'
    lora_path = f'/root/{path}'
    model = PeftModel.from_pretrained(base_model, model_id=lora_path)
    experiment_params = os.path.basename(os.path.dirname(lora_path))
    # experiment_params = experiment_params.split('_seed')[0]
    filename = f"result/{experiment_params}.txt"
    with open(filename, 'a') as file:
        file.write(f"Seed: {seed}\n")
    
    # model = base_model    
    # filename = "result/try.txt"
    # random_seed = 7
    # set_seed(random_seed)
    
    
    
    # for name, param in model.named_parameters():
    #     if param.dtype != torch.bfloat16:
    #         param.data = param.data.to(torch_dtype)
    #         # print(f"Converted {name} to {torch.bfloat16}")
            
    # invalid_params = []
    # for name, parameters in model.named_parameters():
    #     # logger.info(f"{name}, : {parameters.dtype}")
    #     if parameters.dtype != torch_dtype:
    #         invalid_params.append((name, parameters.dtype))
    
    # if invalid_params:
    #     for name, dtype in invalid_params:
    #         print(f"Parameter {name} is not of type {torch_dtype}, but {dtype}.")
        # sys.exit(1)
    
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = model.device
    # model.to(device)
    model.eval()
    # print(model)
    print('Model loaded')
    
    
    batch_size = 32
    batch_test_mt(model, test_ds, tokenizer, device, batch_size, filename)
    # test_math(model, test_ds, tokenizer, device, batch_size, filename)
