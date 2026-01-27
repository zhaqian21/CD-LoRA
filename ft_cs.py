import logging
import math
import os
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import datasets
from datasets import Dataset, DatasetDict, concatenate_datasets
from build_dataset import build_instruction_dataset, DataCollatorForSupervisedDataset
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict
from peft.tuners.lora import LoraLayer

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from custom import *
from parameter import *
from reinit import *
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # print("get parser")
    # # 输出显示参数
    # print("Model Arguments:", model_args)
    # print("Data Training Arguments:", data_args)
    # print("Training Arguments:", training_args)
    if training_args.flash_attn:
        from flash_attn_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)


    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # transformers.tokenization_utils.logging.set_verbosity_warning()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print('last_checkpoint',last_checkpoint)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }
    
    if model_args.tokenizer_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)
        # print("load tokenizer")
        # print(tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    logger.info(f"torch_dtype: {torch_dtype}")
    logger.info(f"bf16: {training_args.bf16}")
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))


    device_map = {"":int(os.environ.get("LOCAL_RANK") or 0)}
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        # config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch_dtype,
        # low_cpu_mem_usage=True,
        # device_map=device_map,
    )
    model.enable_input_require_grads()
    # print(model)

    # if training_args.load_in_kbits in [4, 8]:
    #     model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    # model.config.use_cache = False

    # model_vocab_size = model.get_input_embeddings().weight.shape[0]
    # logger.info(f"Model vocab size: {model_vocab_size}")
    # logger.info(f"len(tokenizer):{len(tokenizer)}")
    # if model_vocab_size != len(tokenizer):
    #     logger.info(f"Resize model vocab size to {len(tokenizer)}")
    #     model.resize_token_embeddings(len(tokenizer))

    if training_args.peft_path is not None: # --------------------------> train from the trained lora model
        logger.info("Peft from pre-trained model")

        model = PeftModel.from_pretrained(model, training_args.peft_path,
            # device_map=device_map
            )
    else: # --------------------------> train from the sketch
        logger.info("Init new peft model") 
        target_modules = training_args.trainable.split(',') # lora paras
        modules_to_save = training_args.modules_to_save # not lora paras, but is trainable, i.e., not freeze
        if modules_to_save is not None:
            modules_to_save = modules_to_save.split(',')
        lora_rank = training_args.lora_rank
        lora_dropout = training_args.lora_dropout
        lora_alpha = training_args.lora_alpha
        
        lora_nums = training_args.lora_nums
        
        
        logger.info(f"target_modules: {target_modules}")
        logger.info(f"lora_rank: {lora_rank}")
        logger.info(f"lora_nums: {lora_nums}")


        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=lora_rank, 
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_nums=lora_nums,
            modules_to_save=modules_to_save
            )
        
        model = get_peft_model(model, peft_config)

    if training_args.gradient_checkpointing and \
        (not model.modules_to_save or 'embed_tokens' not in model.modules_to_save):
        # enable requires_grad to avoid exception during backward pass when using gradient_checkpoint without tuning embed.
        if hasattr(model.base_model, "enable_input_require_grads"):
            model.base_model.enable_input_require_grads()
        elif hasattr(model.base_model, "get_input_embeddings"):
            def make_inputs_require_grad(_module, _input, _output):
                _output.requires_grad_(True)
            model.base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # 精度设置
    # for name, module in model.named_modules():
    #     if isinstance(module, LoraLayer):
    #         if training_args.bf16:
    #             module = module.to(torch.bfloat16)
    #             # 确保 lora_route 也转换为 bfloat16
    #             if hasattr(module, 'lora_route'):
    #                 module.lora_route.weight.data = module.lora_route.weight.data.to(torch.bfloat16)
    #                 if module.lora_route.bias is not None:
    #                     module.lora_route.bias.data = module.lora_route.bias.data.to(torch.bfloat16)
    #         if training_args.fp16:
    #             module = module.to(torch.float16)
    #     if 'norm' in name:
    #         module = module.to(torch.float16)
    #     if 'lm_head' in name or 'embed_tokens' in name:
    #         if hasattr(module, 'weight'):
    #             if training_args.bf16 and module.weight.dtype == torch.float32:
    #                 module = module.to(torch.bfloat16)
    #             if training_args.fp16 and module.weight.dtype == torch.float32:
    #                 module = module.to(torch.float16)
    
    
    # # 合并所有训练集和验证集
    # train_datasets = [CustomDataset2(dataset) for dataset in train_ds.values()]

    # glue_token_dir = "/root/autodl-tmp/data/glue_tokenized"
    # # DATASET = ["sst2", "qqp", "cola", "qnli"]
    # DATASET = ["sst2", "qqp", "qnli"]
    # glue_train_ds, glue_test_ds = load_tokenized_datasets(glue_token_dir, DATASET)
    
    # DATASET = ["piqa", "siqa"]
    # # DATASET = ["boolq", "piqa", "siqa"]
    # cs_token_dir = "/root/autodl-tmp/data/commonsense_tokenized"
    # cs_train_ds, cs_test_ds = load_tokenized_datasets(cs_token_dir, DATASET)
       
    # train_ds = {**glue_train_ds, **cs_train_ds}
    # # train_ds = glue_train_ds

    # # check_random_seeds()

    # if data_args.sample:
    #     sample_size = data_args.sample_size
    #     for key in train_ds.keys():
    #         if len(train_ds[key]) > sample_size:
    #             indices = random.sample(range(len(train_ds[key])), sample_size)
    #             train_ds[key] = train_ds[key].select(indices)     
        
    # for name, dataset in train_ds.items():
    #     print(name)
    #     # print(dataset[0])
    #     print(dataset)

    # logger.info(f"Data construction: {data_args.data_mix}")
    # if data_args.data_mix == "mix":
    #     train_datasets = [CustomDataset2(dataset) for dataset in train_ds.values()]
    #     ratios = [1.0 / len(train_datasets)] * len(train_datasets)  # 均匀采样
    #     # ratios=[0.1, 0.2, 0.2, 0.2, 0.3]
    #     print(ratios)
    #     sampler = MultiDatasetSampler(train_datasets, training_args.per_device_train_batch_size, ratios)
    #     print(sampler)
    #     print(len(sampler))
    #     # 使用 ConcatDataset 合并多个数据集
    #     concat_dataset = ConcatDataset(train_datasets)
    #     print(concat_dataset)
    #     print(len(concat_dataset))
    #     # 创建 DataLoader
    #     train_dataloader = DataLoader(
    #         concat_dataset,
    #         batch_size=training_args.per_device_train_batch_size,
    #         sampler=sampler,
    #         collate_fn=custom_collate_fn,
    #         drop_last=True
    #     )
    #     # 检查数据
    #     for batch in train_dataloader:
    #         print(batch.keys())  # 打印 batch 的键
    #         print(batch['input_ids'].shape)  # 打印 input_ids 的形状
    #         print(batch['attention_mask'].shape)  # 打印 attention_mask 的形状
    #         print(batch['labels'].shape)  # 打印 labels 的形状
    #         # 将 input_ids 解码为文本
    #         input_texts = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
    #         # label_texts = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
    #         print("解码后的文本:")
    #         for i, text in enumerate(input_texts):
    #             print(f"样本 {i + 1}: {text}")
    #         break  # 只检查第一个 batch
    # else:
    #     train_datasets = [dataset for dataset in train_ds.values()]
    #     train_datasets = concatenate_datasets(train_datasets)
    #     if data_args.data_mix == "random":
    #         train_datasets = train_datasets.shuffle()
    
    # train_datasets = Dataset.from_dict(train_datasets)
    # train_datasets = train_datasets[::-1]


 
    # logger.info(f"Num train_samples  {len(train_datasets)}")
    # logger.info(f"len(train_dataloader): {len(train_dataloader)}")


    
    # 合并成一个训练集和验证集
    # train_dataset = concatenate_datasets(train_datasets)
    # eval_dataset = concatenate_datasets(sampled_eval_datasets)
    # logger.info(f"Num train_samples  {len(train_dataset)}")
    # logger.info(f"Num eval_samples  {len(eval_dataset)}")

    train_datasets = load_data()

    if model_args.reinit:
        logger.info("Reinit lora layer")
        if model_args.init_mode == "gradient":
            init_bs = model_args.init_bs
            named_grads_list = []
            if isinstance(train_ds, DatasetDict):
                for subset_name in train_ds.keys():
                    subset = train_ds[subset_name]  # 获取子数据集
                    indices = random.sample(range(len(subset)), sample_size)
                    subset = subset.select(indices)
                    # 估计梯度
                    named_grads = estimate_gradient(model, subset, init_bs)
                    # 将梯度存入列表
                    named_grads_list.append(named_grads)
            elif isinstance(train_ds, Dataset):
                # 如果 train_ds 是单个数据集，直接采样并估计梯度
                indices = random.sample(range(len(train_ds)), sample_size)
                subset = train_ds.select(indices)
                named_grads = estimate_gradient(model, subset, init_bs)
                named_grads_list.append(named_grads)
            else:
                raise ValueError("train_ds 必须是 Hugging Face 的 DatasetDict 或 Dataset 对象")
           
            reinit_lora(model, model_args, named_grads_list)
        else:
            reinit_lora(model, model_args)

    model.print_trainable_parameters()
    logger.info(f"model.modules_to_save: {model.modules_to_save}")
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))


    # invalid_params = []
    # for name, parameters in model.named_parameters():
    #     # logger.info(f"{name}, : {parameters.dtype}")
    #     if parameters.dtype != torch_dtype:
    #         invalid_params.append((name, parameters.dtype))

    # if invalid_params:
    #     for name, dtype in invalid_params:
    #         print(f"Parameter {name} is not of type {torch_dtype}, but {dtype}.")
    #     # sys.exit(1)

    training_args.remove_unused_columns = False
    
    # Initialize our Trainer
    if data_args.data_mix == "mix":
        trainer = CustomTrainer_data(
            model=model,
            args=training_args,
            train_dataset=concat_dataset,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            # data_collator=data_collator
        )
        trainer.add_callback(SavePeftModelCallback(tokenizer=tokenizer))
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_datasets,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        trainer.add_callback(SavePeftModelCallback(tokenizer=tokenizer))

    logger.info(f"Model default dtype: {model.dtype}")
    
    check_random_seeds()
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics

        # metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
