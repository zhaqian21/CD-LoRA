import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import datasets
import torch
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

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")


class SavePeftModelCallback(transformers.TrainerCallback):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "sft_lora_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "sft_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        # kwargs["tokenizer"].save_pretrained(peft_model_path)
        # self.tokenizer.save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "sft_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        # kwargs["tokenizer"].save_pretrained(peft_model_path)


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # cast all non INT8/INT4 parameters to fp32
    for param in model.parameters():
        if ((param.dtype == torch.float16) or (param.dtype == torch.bfloat16)) and loaded_in_kbit:
            param.data = param.data.to(torch.float32)

    for name, module in model.named_modules():
        if 'norm' in name:
            module = module.to(torch.float32)

    if loaded_in_kbit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, _input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="/root/autodl-tmp/qwen/Qwen2.5-0.5B-Instruct",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default="/root/autodl-tmp/qwen/Qwen2.5-0.5B-Instruct",
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to allow to load models from a custom codebase."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    
    
    reinit: bool = field(
        default=False,
        metadata={"help": "Whether to reinit the lora layer"},
    )
    rank_stablization: bool = field(
        default=False,
        metadata={"help": "Whether to use rank stablization for scaling"},
    )    
    lora_A_init: Optional[str] = field(
        default="unit",
        metadata={"help": "Initialization method for lora_A",
        "choices": ["gaussian", "kaiming", "fan_out_kaiming", "xavier", "zeros", "unit", "orthogonal"]
        }
    )
    lora_B_init: Optional[str] = field(
        default="unit",
        metadata={"help": "Initialization method for lora_B",
        "choices": ["gaussian", "kaiming", "fan_out_kaiming", "xavier", "zeros", "unit", "orthogonal"]
        }
    )
    init_mode: Optional[str] = field(
        default="simple",
        metadata={"help": "Initialization mode: simple or svd"}
    )
    init_scale: Optional[str] = field(
        default="stable",
        metadata={"help": "Scaling method for initialization"}
    )
    stable_gamma: Optional[int] = field(
        default=64,
        metadata={"help": "Gamma value for stable scaling"}
    )
    init_bs: Optional[int] = field(
        default=5,
        metadata={"help": "Batch size for gradient initialization"}
    )
    # lora_A_std: Optional[float] = field(
    #     default=0.02,
    #     metadata={"help": "Standard deviation for gaussian initialization of lora_A"}
    # )
    # lora_B_std: Optional[float] = field(
    #     default=0.02,
    #     metadata={"help": "Standard deviation for gaussian initialization of lora_B"}
    # )


    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    processed_dir: Optional[str] = field(
        default=None, metadata={"help": "Pre-procesed data dir."}
    )

    tokenized_dir: Optional[str] = field(
        default=None, metadata={"help": "Tokenized data dir."}
    )

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    data_cache_dir: Optional[str] = field(default=None, metadata={"help": "The datasets processed stored"})

    max_seq_length: Optional[int] = field(default=1024)


@dataclass
class MyTrainingArguments(TrainingArguments):
    trainable : Optional[str] = field(default="q_proj,v_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.1)
    lora_alpha : Optional[float] = field(default=32.)
    modules_to_save : Optional[str] = field(default=None)
    peft_path : Optional[str] = field(default=None)
    flash_attn : Optional[bool] = field(default=False)
    double_quant: Optional[bool] = field(default=True)
    quant_type: Optional[str] = field(default="nf4")
    load_in_kbits: Optional[int] = field(default=16)
    bf16: Optional[bool] = field(default=True)
    lora_nums: Optional[int] = field(default=2)



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

    # if (len(tokenizer)) != 55296:
    #     raise ValueError(f"The vocab size of the tokenizer should be 55296, but found {len(tokenizer)}.\n"
    #                      "Please use Chinese-LLaMA-2 tokenizer.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    eval_dataset=None
    train_dataset = None

    # if training_args.do_train:
    #     with training_args.main_process_first(desc="loading and tokenization"):
    #         path = Path(data_args.dataset_dir)
    #         files = [os.path.join(path,file.name) for file in path.glob("*.json")]
    #         logger.info(f"Training files: {' '.join(files)}")
    #         train_dataset = build_instruction_dataset(
    #             data_path=files,
    #             tokenizer=tokenizer,
    #             max_seq_length=data_args.max_seq_length,
    #             data_cache_dir = None,
    #             preprocessing_num_workers = data_args.preprocessing_num_workers)
    #     logger.info(f"Num train_samples  {len(train_dataset)}")
    #     logger.info(f"Training example input: {tokenizer.decode(train_dataset[0]['input_ids'])}")
    #     logger.info(f"Training example: {train_dataset[0]}")
    
    # if training_args.do_eval:
    #     with training_args.main_process_first(desc="loading and tokenization"):
    #         files = [data_args.validation_file]
    #         logger.info(f"Evaluation files: {' '.join(files)}")
    #         eval_dataset = build_instruction_dataset(
    #             data_path=files,
    #             tokenizer=tokenizer,
    #             max_seq_length=data_args.max_seq_length,
    #             data_cache_dir = None,
    #             preprocessing_num_workers = data_args.preprocessing_num_workers)
    #     logger.info(f"Num eval_samples  {len(eval_dataset)}")
    #     logger.info(f"Evaluation example input: {tokenizer.decode(eval_dataset[0]['input_ids'])}")
    #     logger.info(f"Evaluation example: {eval_dataset[0]}")

    tokenized_dir = data_args.tokenized_dir
    logger.info(f"Tokenized dir: {tokenized_dir}")
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

    ratios = [1.0 / len(train_datasets)] * len(train_datasets)  # 均匀采样
    print(ratios)
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
    logger.info(f"Num train_samples  {len(concat_dataset)}")
    logger.info(f"len(train_dataloader): {len(train_dataloader)}")

    # 检查数据
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
        break  # 只检查第一个 batch
    
    # 合并成一个训练集和验证集
    # train_dataset = concatenate_datasets(train_datasets)
    # eval_dataset = concatenate_datasets(sampled_eval_datasets)
    # logger.info(f"Num train_samples  {len(train_dataset)}")
    # logger.info(f"Num eval_samples  {len(eval_dataset)}")
    
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

    if training_args.load_in_kbits in [4, 8]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    model.config.use_cache = False

    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    logger.info(f"Model vocab size: {model_vocab_size}")
    logger.info(f"len(tokenizer):{len(tokenizer)}")
    if model_vocab_size != len(tokenizer):
        logger.info(f"Resize model vocab size to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

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
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
            if training_args.fp16:
                module = module.to(torch.float16)
        if 'norm' in name:
            module = module.to(torch.float16)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
                if training_args.fp16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.float16)
    
    if model_args.reinit:
        logger.info("Reinit lora layer")
        if model_args.init_mode == "gradient":
            if isinstance(train_ds, DatasetDict):
                for subset_name in train_ds.keys():
                    subset = train_ds[subset_name]  # 获取子数据集
                    # 随机采样 init_bs 个数据
                    sampled_subset = subset.shuffle(seed=42).select(range(init_bs))
                    # 估计梯度
                    named_grads = estimate_gradient(model, sampled_subset, init_bs)
                    # 将梯度存入列表
                    named_grads_list.append(named_grads)
            elif isinstance(train_ds, Dataset):
                # 如果 train_ds 是单个数据集，直接采样并估计梯度
                sampled_subset = train_ds.shuffle(seed=42).select(range(init_bs))
                named_grads = estimate_gradient(model, sampled_subset, init_bs)
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


    for name, parameters in model.named_parameters():
        # logger.info(f"{name}, :, {parameters.size()},{parameters.requires_grad}")
        logger.info(f"{name}, : {parameters.dtype}")


    training_args.remove_unused_columns = False
    # Initialize our Trainer
    trainer = CustomTrainer_data(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        # data_collator=data_collator
    )
    trainer.add_callback(SavePeftModelCallback(tokenizer=tokenizer))
    
    # trainer = Trainer(
    # model=model,
    # args=training_args,
    # train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
    # data_collator=data_collator,
    # callbacks=[SavePeftModelCallback(tokenizer=tokenizer)],
    # tokenizer=tokenizer,
    # )
    logger.info(f"Model default dtype: {model.dtype}")
    
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
