import torch
import argparse
import random
from functools import partial
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset, get_dataset_config_names, concatenate_datasets
from peft import LoraConfig, TaskType, PeftModel, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    GenerationConfig,
    set_seed,
)
from data_load import load_data, check_random_seeds  # 假设你已经实现了 load_data 函数
from custom import *  # 假设你有自定义的模块

# 设置随机种子
def set_random_seed(seed):
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


# 主函数
def main(seed, output_dir):
    print(seed)
    # 设置随机种子
    set_random_seed(seed)
    # set_seed(seed)

    # 加载模型和分词器
    model_base = AutoModelForCausalLM.from_pretrained(
        "/root/autodl-tmp/qwen/Qwen2.5-3B",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        output_hidden_states=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "/root/autodl-tmp/qwen/Qwen2.5-3B",
        use_fast=False,
        trust_remote_code=True,
    )

    # 开启梯度检查点
    model_base.enable_input_require_grads()

    # 配置 LoRA
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=10,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model_base, config)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,  # 使用传入的 output_dir
        logging_dir=f"{output_dir}/logs",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        logging_steps=100,
        num_train_epochs=1,
        save_steps=500,
        save_total_limit=3,
        learning_rate=2e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
    )

    # 打印可训练参数
    model.print_trainable_parameters()

    # 加载数据集
    train_dataset = load_data()

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    print(model.dtype)
    print(model.device)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model.to(device)
    
    set_random_seed(seed)

    # 检查随机种子
    check_random_seeds()
    model.print_trainable_parameters()
    # 开始训练
    trainer.train()

# 命令行参数解析
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for training results")
    args = parser.parse_args()

    # 调用主函数
    main(args.seed, args.output_dir)