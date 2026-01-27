import torch
import random
from functools import partial
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset, get_dataset_config_names, concatenate_datasets
from peft import LoraConfig, TaskType, PeftModel, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from data_load import *
from custom import *

model_base = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/qwen/Qwen2.5-0.5B-Instruct',
                                              device_map="auto",
                                              torch_dtype=torch.bfloat16,
                                              output_hidden_states=True
                                              )
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/qwen/Qwen2.5-0.5B-Instruct', 
                                          use_fast=False, trust_remote_code=True)

print(model_base)
model_base.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
print(model_base.dtype)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=16, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
model = get_peft_model(model_base, config)
print(config)

training_args = TrainingArguments(
    output_dir="./output/Qwen2.5_0.5B_glue_mkmmd_3_lora",
    logging_dir="./logs/Qwen2.5_0.5B_glue_mkmmd_3_lora",
    per_device_train_batch_size=10,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=1,
    save_steps=100, 
    learning_rate=2e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    # remove_unused_columns=False,
)

tokenized_dir = "/root/autodl-tmp/data/glue_tokenized"
train_ds, test_ds = load_tokenized_datasets(tokenized_dir)

# 获取训练数据集列表
train_datasets = [CustomDataset2(dataset) for dataset in train_ds.values()]
# train_datasets = [dataset for dataset in train_ds.values()]

# 定义 batch_size 和采样比例
batch_size = 10
ratios = [1.0 / len(train_datasets)] * len(train_datasets)  # 均匀采样
print(ratios)

# 创建自定义 Sampler
sampler = MultiDatasetSampler(train_datasets, training_args.per_device_train_batch_size, ratios)
print(sampler)
print(len(sampler))

# 使用 ConcatDataset 合并多个数据集
concat_dataset = ConcatDataset(train_datasets)
print(concat_dataset)
print(len(concat_dataset))

# 创建 DataLoader
train_dataloader = DataLoader(
    concat_dataset,
    batch_size=training_args.per_device_train_batch_size,
    sampler=sampler,
    collate_fn=custom_collate_fn,
    drop_last=True
)

# 验证实现
print(len(train_dataloader))

for batch in train_dataloader:
    print(batch.keys())  # 打印 batch 的键
    print(batch['input_ids'].shape)  # 打印 input_ids 的形状
    print(batch['attention_mask'].shape)  # 打印 attention_mask 的形状
    print(batch['labels'].shape)  # 打印 labels 的形状
    break  # 只检查第一个 batch

# 创建自定义 Trainer
trainer = CustomTrainer4(
    model=model,
    args=training_args,
    train_dataset=concat_dataset,  # 传递数据集
    custom_train_dataloader=train_dataloader  # 传递自定义的 DataLoader
)
print(trainer)
trainer.train()