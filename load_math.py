import torch
import os
import random
from collections import Counter
from datasets import Dataset, DatasetDict, load_dataset, get_dataset_config_names, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer
from data_load import *
task = "gsm8k"
DATASET = [task]
save_dir = f"/root/autodl-tmp/data/{task}_processed"
test_set= load_gsm8k()

dataset_dict = DatasetDict({
    # "train": train_set,
    # "validation": validation_set,
    "test": test_set
})

dataset_dict.save_to_disk(f"{save_dir}")

dataset = load_from_disk(f"{save_dir}")
test_set = dataset["test"]
print(test_set)
train_ds = {task:train_set}
valid_ds = {task:dataset["validation"]}
test_ds = {task:dataset["test"]}
print(train_ds, valid_ds, test_ds)
加载分词器
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/qwen/Qwen2.5-0.5B-Instruct', 
                                          use_fast=False, trust_remote_code=True)
train_dataset = dataset["train"].map(
    lambda examples: preprocess_fc(examples, tokenizer),
    batched=True,
    batch_size=1000,
    remove_columns=dataset["train"].column_names  # 移除原始列（['query', 'response', 'type', 'original_question', 'x', 'y']）
)
# train_ds, test_ds = tokenized_dataset(train_ds, test_ds, tokenizer)
# for name, dataset in valid_ds.items():
#     valid_ds[name] = dataset.map(
#         lambda examples: preprocess_fc(examples, tokenizer),
#         batched=True,
#         batch_size=100,
#         remove_columns=dataset.column_names
#     )
# # 保存处理后的数据集
# save_tokenized_dir = f"/root/autodl-tmp/data/{task}_tokenized"
# valid_ds[name].save_to_disk(f"{save_tokenized_dir}/{task}_valid")
# print(f"Saved tokenized {task} train dataset to {save_tokenized_dir}/{task}_train")
# save_tokenized_datasets(train_ds, test_ds, save_tokenized_dir)
# # 加载处理后的数据集
# train_ds[task] = load_from_disk(f"{save_tokenized_dir}/{task}_train")
# print(f"Loaded tokenized {task} train dataset from {save_tokenized_dir}/{task}_train")
# tokenized_train_ds, tokenized_test_ds = load_tokenized_datasets(save_tokenized_dir, DATASET)

# for name, dataset in tokenized_train_ds.items():
#     print(name)
#     # print(dataset[0])
#     print(dataset)