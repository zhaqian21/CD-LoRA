from custom import *
from data_load import *
from transformers import AutoTokenizer
tokenized_dir = "/root/autodl-tmp/data/glue_tokenized2"
train_ds, test_ds = load_tokenized_datasets(tokenized_dir)

# 合并所有训练集和验证集
train_datasets = [CustomDataset2(dataset) for dataset in train_ds.values()]
# train_datasets = train_datasets[::-1]

# eval_datasets = [dataset for dataset in test_ds.values()]
# # 检验每个验证集的长度，超过五千条的就采样五千条
# sampled_eval_datasets = [
#     dataset.select(range(5000)) 
#     if len(dataset) > 5000 else dataset for dataset in eval_datasets
# ]
batch_size = 5
ratios = [1.0 / len(train_datasets)] * len(train_datasets)  # 均匀采样
print(ratios)
sampler = MultiDatasetSampler(train_datasets, batch_size, ratios)
# print(sampler)
print(len(sampler))
# 使用 ConcatDataset 合并多个数据集
concat_dataset = ConcatDataset(train_datasets)
# print(concat_dataset)
print(len(concat_dataset))
# 创建 DataLoader
train_dataloader = DataLoader(
    concat_dataset,
    batch_size=batch_size,
    sampler=sampler,
    collate_fn=custom_collate_fn,
    drop_last=True
)
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/qwen/Qwen2.5-0.5B-Instruct', 
                                          use_fast=False, trust_remote_code=True)
print("get batch")
for batch in train_dataloader:
    print(batch.keys())  # 打印 batch 的键
    print(batch['input_ids'].shape)  # 打印 input_ids 的形状
    print(batch['attention_mask'].shape)  # 打印 attention_mask 的形状
    print(batch['labels'].shape)  # 打印 labels 的形状
        # 将 input_ids 解码为文本
    input_texts = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
    # label_texts = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
    print("解码后的文本:")
    for i, text in enumerate(input_texts):
        print(f"样本 {i + 1}: {text}")

    print(batch['labels'].tolist())
    break  # 只检查第一个 batch

