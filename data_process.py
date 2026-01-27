import torch
import os
import random
from collections import Counter
from datasets import Dataset, DatasetDict, load_dataset, get_dataset_config_names, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer
from data_load import *

save_dir = "/root/autodl-tmp/data/glue_processed"
# save_glue_disk(save_dir)
DATASET = ["sst2", "cola", "qqp", "mnli", "qnli",]
# train_ds, test_ds = load_ds_from_disk(save_dir, DATASET)

# # 加载分词器
# tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/qwen/Qwen2.5-0.5B-Instruct', 
#                                           use_fast=False, trust_remote_code=True)

# train_ds, test_ds = tokenized_dataset(train_ds, test_ds, tokenizer)

# # 保存处理后的数据集
# save_tokenized_dir = "/root/autodl-tmp/data/glue_tokenized"
# save_tokenized_datasets(train_ds, test_ds, save_tokenized_dir)

# 加载处理后的数据集
save_tokenized_dir = "/root/autodl-tmp/data/glue_tokenized"
tokenized_train_ds, tokenized_test_ds = load_tokenized_datasets(save_tokenized_dir, DATASET)

for name, dataset in tokenized_train_ds.items():
    print(name)
    # print(dataset[0])
    print(dataset)