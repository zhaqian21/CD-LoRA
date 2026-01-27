from data_load import *
import torch
import random
from datasets import Dataset, load_dataset, get_dataset_config_names, concatenate_datasets, interleave_datasets
from peft import LoraConfig, TaskType, PeftModel, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


print('Loading model...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/qwen/Qwen2.5-0.5B-Instruct', 
                                          use_fast=False, trust_remote_code=True)
model_base = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/qwen/Qwen2.5-0.5B-Instruct',
                                              device_map="auto",
                                              torch_dtype=torch.float32,
                                              output_hidden_states=True
                                              )
lora_path = '/root/autodl-tmp/HydraLoRA/HydraLoRA/output/Qwen2.5-0.5B-Instruct-glue-multi_dropout0.1_mix/checkpoint-8500'
model = PeftModel.from_pretrained(model_base, model_id=lora_path)
model.to(device)
model.eval()
print(model)
print('Model loaded')


tokenized_dir = "/root/autodl-tmp/data/glue_processed"
train_ds, test_ds = load_glue_from_disk(tokenized_dir)                                        

random.seed(2)
sample_size = 5000
for key in test_ds.keys():
    if len(test_ds[key]) > sample_size:
        indices = random.sample(range(len(test_ds[key])), sample_size)
        test_ds[key] = test_ds[key].select(indices)

# 检查所有数据集的标签分布
for name, dataset in test_ds.items():
    print(name)
    # print(dataset[0])
    print(dataset)
    check_label_distribution(dataset, f"{name} 测试集")

filename="results_multi_mix_2.txt"

for name, dataset in test_ds.items():
    print(name)
    with open(filename, "a") as file:
        # 将精度写入文件，并添加换行符
        file.write(f"Task {name}\n")
    correct_count = 0  # 正确预测的数量
    total_count = 0
    # samples = test_ds[name].select(range(2))
    idx = 0
    for sample in dataset:
        label = sample['y']
        input_text = sample['x']
        # print(type(label), type(input_text))
        # print("input:", input_text)
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']  # 显式传递 attention_mask
        )
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print("output:",pred_text)
        pred = parse_model_output(pred_text)
        # print("pred:",pred)
        # # print(input)
        # print("label:",label)
        if pred == label:
            correct_count += 1
        total_count += 1
        if (idx+1)%100 == 0:
            accuracy = correct_count / total_count
            print(f"{idx} Accuracy: {accuracy:.2%}")
            with open(filename, "a") as file:
                # 将精度写入文件，并添加换行符
                file.write(f"{idx} Accuracy: {accuracy:.2%}\n")
        idx += 1
    accuracy = correct_count / total_count
    print(f"{idx} Accuracy: {accuracy:.2%}")
    with open(filename, "a") as file:
        # 将精度写入文件，并添加换行符
        file.write(f"{idx} Accuracy: {accuracy:.2%}\n")
    print("===")