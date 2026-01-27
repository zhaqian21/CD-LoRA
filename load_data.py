import torch
import os
import random
from collections import Counter
from datasets import Dataset, DatasetDict, load_dataset, get_dataset_config_names, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer
from data_load import *

SEED = 42
ran = random.Random(SEED)
SAMPLE_SIZE = {"sst2": 20000, "qqp": 20000, "mnli": 10000, "qnli": 10000,
               "boolq": 10000, "obqa": 5000, "piqa": 10000, "siqa": 10000, "winogrande": 20000,
               "metamathqa":20000, "cosmosqa":30000, "commongen":10000,
                "dart":10000, "arc": 10000, "nq": 13000}
glue_dir = "/root/autodl-tmp/data/glue_tokenized"
DATASET = ["sst2","qqp", "mnli", "qnli",]
glue_ds ={task: None for task in DATASET}
for task_name in DATASET:
    # 加载处理后的训练集
    train_set = load_from_disk(f"{glue_dir}/{task_name}_train")
    if len(train_set) > SAMPLE_SIZE[task_name]:        
        indices = ran.sample(range(len(train_set)), SAMPLE_SIZE[task_name])
        train_set = train_set.select(indices)
    glue_ds[task_name] = train_set  
    print(f"Loaded tokenized {task_name} train dataset from {glue_dir}/{task_name}_train")

cs_dir = "/root/autodl-tmp/data/commonsense_tokenized"
DATASET = ["boolq","piqa", "siqa", "winogrande"]
cs_ds = {task: None for task in DATASET}
for task_name in DATASET:
    # 加载处理后的训练集
    train_set = load_from_disk(f"{cs_dir}/{task_name}_train")
    if len(train_set) > SAMPLE_SIZE[task_name]:
        indices = ran.sample(range(len(train_set)), SAMPLE_SIZE[task_name])
        train_set = train_set.select(indices)
    cs_ds[task_name] = train_set  
    print(f"Loaded tokenized {task_name} train dataset from {cs_dir}/{task_name}_train")

obqa_train_ds = load_from_disk("/root/autodl-tmp/data/obqa_tokenized/obqa_train")
obqa_test_ds = load_from_disk("/root/autodl-tmp/data/obqa_tokenized/obqa_test")
obqa_ds = concatenate_datasets([obqa_train_ds, obqa_test_ds])
obqa_ds = {"obqa": obqa_ds}
print(f"Loaded tokenized obqa train dataset from /root/autodl-tmp/data/obqa_tokenized/obqa_train")

cos_ds = load_from_disk("/root/autodl-tmp/data/cosmosqa_tokenized/cosmosqa_train")
if len(cos_ds) > SAMPLE_SIZE["cosmosqa"]:
    indices = ran.sample(range(len(cos_ds)), SAMPLE_SIZE["cosmosqa"])
    cos_ds = cos_ds.select(indices)
cos_ds = {"cosmosqa": cos_ds}
print(f"Loaded tokenized cosmosqa train dataset from /root/autodl-tmp/data/cosmosqa_tokenized/cosmosqa_train")

commongen_ds = load_from_disk("/root/autodl-tmp/data/commongen_tokenized/commongen_train")
if len(commongen_ds) > SAMPLE_SIZE["commongen"]:
    indices = ran.sample(range(len(commongen_ds)), SAMPLE_SIZE["commongen"])
    commongen_ds = commongen_ds.select(indices)
commongen_ds = {"commongen": commongen_ds}
print(f"Loaded tokenized commongen train dataset from /root/autodl-tmp/data/commongen_tokenized/commongen_train")

dart_ds = load_from_disk("/root/autodl-tmp/data/dart_tokenized/dart_train")
if len(dart_ds) > SAMPLE_SIZE["dart"]:
    indices = ran.sample(range(len(dart_ds)), SAMPLE_SIZE["dart"])
    dart_ds = dart_ds.select(indices)
dart_ds = {"dart": dart_ds}
print(f"Loaded tokenized dart train dataset from /root/autodl-tmp/data/dart_tokenized/dart_train")

arc_train_ds = load_from_disk("/root/autodl-tmp/data/arc_tokenized/arc_train")
arc_test_ds = load_from_disk("/root/autodl-tmp/data/arc_tokenized/arc_test")
arc_ds = concatenate_datasets([arc_train_ds, arc_test_ds])
arc_ds = {"arc": arc_ds}
print(f"Loaded tokenized arc train dataset from /root/autodl-tmp/data/arc_tokenized/arc_train")

metamathqa_ds = load_from_disk("/root/autodl-tmp/data/metamathqa_tokenized/metamathqa_train")
if len(metamathqa_ds) > SAMPLE_SIZE["metamathqa"]:
    indices = ran.sample(range(len(metamathqa_ds)), SAMPLE_SIZE["metamathqa"])
    metamathqa_ds = metamathqa_ds.select(indices)
metamathqa_ds = {"metamathqa": metamathqa_ds}
print(f"Loaded tokenized metamathqa train dataset from /root/autodl-tmp/data/metamathqa_tokenized/metamathqa_train")

nq_ds = load_from_disk("/root/autodl-tmp/data/nq_tokenized/nq_train")
if len(nq_ds) > SAMPLE_SIZE["nq"]:
    indices = ran.sample(range(len(nq_ds)), SAMPLE_SIZE["nq"])
    nq_ds = nq_ds.select(indices)
nq_ds = {"nq": nq_ds}
print(f"Loaded tokenized nq train dataset from /root/autodl-tmp/data/nq_tokenized/nq_train")

train_ds = {**glue_ds, **cs_ds, **obqa_ds, **cos_ds, **commongen_ds, **dart_ds,**metamathqa_ds,  **arc_ds, **nq_ds}

for name, dataset in train_ds.items():
    print(name)
    print(dataset)


train_datasets = concatenate_datasets([dataset for dataset in train_ds.values()])
train_datasets = train_datasets.shuffle(seed=SEED)
print(train_datasets)