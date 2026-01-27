import logging
import math
import os
import sys
import torch
import datasets
import torch.nn.functional as F
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

from peft import LoraConfig,TaskType, get_peft_model, PeftModel, get_peft_model_state_dict
# from peft.tuners.rlora import RLoraConfig, RLoraModel
from data_load import *
from parameter import *
from reinit import *
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


class RDropTrainer(Trainer):
    """
    Trainer with R-Drop regularization.
    It performs two forward passes with dropout on and adds a symmetric KL term between
    the two output distributions.
    """

    # accept extra kwargs like `num_items_in_batch` for compatibility with Trainer
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels", None)
        rdrop_alpha = getattr(self.args, "rdrop_alpha", 0.0)
        if labels is None or rdrop_alpha <= 0.0:
            # 不启用 R-Drop 时，直接走原始实现
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        # 单次前向：KL 已在 LoRA A 层内部计算为 module.rdrop_loss
        outputs = model(**inputs)
        logits = outputs.logits

        # 标准 LM 交叉熵
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()

        vocab_size = shift_logits.size(-1)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        ce_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))

        # 收集各个 LoRA 层上累积的 KL 损失
        rdrop_loss = 0.0
        for module in model.modules():
            if hasattr(module, "rdrop_loss"):
                rdrop_loss = rdrop_loss + module.rdrop_loss
                module.rdrop_loss = 0.0  # 清空，避免跨 batch 累加

        loss = ce_loss + rdrop_loss

        if return_outputs:
            return loss, outputs
        return loss

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
            use_router=training_args.use_router,
            modules_to_save=modules_to_save
            )
        
        model = get_peft_model(model, peft_config)
        
    rdrop_alpha = getattr(training_args, "rdrop_alpha", 0.0)
    if rdrop_alpha > 0.0:
        for module in model.modules():
            # 只给带 lora_A 的 LoRA 线性层加属性，避免影响其它模块
            if hasattr(module, "lora_A") and hasattr(module, "lora_num"):
                module.rdrop_alpha = rdrop_alpha
    
    if training_args.gradient_checkpointing and \
        (not model.modules_to_save or 'embed_tokens' not in model.modules_to_save):
        # enable requires_grad to avoid exception during backward pass when using gradient_checkpoint without tuning embed.
        if hasattr(model.base_model, "enable_input_require_grads"):
            model.base_model.enable_input_require_grads()
        elif hasattr(model.base_model, "get_input_embeddings"):
            def make_inputs_require_grad(_module, _input, _output):
                _output.requires_grad_(True)
            model.base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # logger.info("Loading data")
    
    # train_datasets = load_data()
    # logger.info(f"Data construction: {data_args.data_mix}")
    # if data_args.data_mix == "random":
    #     train_datasets = train_datasets.shuffle()
    # task_name = "siqa"
    # save_dir = "/root/autodl-tmp/data/commonsense_tokenized"
    # train_datasets = load_from_disk(f"{save_dir}/{task_name}_train")
    # print(f"Loaded tokenized {task_name} train dataset from {save_dir}/{task_name}_train")


    #eight tasks
    glue_token_dir = "/root/autodl-tmp/data/glue_tokenized"
    DATASET = ["qnli", "cola", "mnli", "qqp", "sst2"]
    glue_train_ds = load_tokenized_datasets(glue_token_dir, DATASET)
    
    # DATASET = ["piqa", "winogrande"]
    DATASET = ["piqa", "winogrande", "siqa", "boolq", "hellaswag"]


    # # DATASET = ["piqa", "siqa"]



    cs_token_dir = "/root/autodl-tmp/data/commonsense_tokenized"
    cs_train_ds = load_tokenized_datasets(cs_token_dir, DATASET)


    
    gsm_train_ds = load_from_disk("/root/autodl-tmp/data/gsm8k_tokenized/gsm8k_train")
    gsm_train_ds = {"gsm8k": gsm_train_ds}

    train_ds = {"gsm8k": gsm_train_ds}


    arc_train_ds = load_from_disk("/root/autodl-tmp/data/arc_tokenized/arc_train")
    arc_test_ds = load_from_disk("/root/autodl-tmp/data/arc_tokenized/arc_test")
    arc_ds = concatenate_datasets([arc_train_ds, arc_test_ds])
    arc_ds = {"arc": arc_ds}


    train_ds = {**glue_train_ds, **cs_train_ds, ** arc_ds, **gsm_train_ds}
    # train_ds = {**glue_train_ds, **cs_train_ds, **arc_ds, **gsm_train_ds}
    # # train_ds = {**gsm_train_ds}
    # # train_ds = arc_ds

    # # ===== 使用 flanv2_tokenized 作为训练数据 =====
    # flan_path = "/root/autodl-tmp/data/flanv2_tokenized"
    # flan_ds = load_from_disk(flan_path)
    # # 如果是 DatasetDict，优先取 train split；否则直接用整个 Dataset
    # if isinstance(flan_ds, dict) and "train" in flan_ds:
    #     flan_train = flan_ds["train"]
    # else:
    #     try:
    #         flan_train = flan_ds["train"]
    #     except Exception:
    #         flan_train = flan_ds
    # train_ds = {"flanv2": flan_train}

    set_seed(41)
    if data_args.sample:
        sample_size = data_args.sample_size
        for key in train_ds.keys():
            if len(train_ds[key]) > sample_size:
                indices = random.sample(range(len(train_ds[key])), sample_size)
                train_ds[key] = train_ds[key].select(indices)
    
    for name, dataset in train_ds.items():
        print(name)
        # print(dataset[0])
        print(dataset)
    train_datasets = [dataset for dataset in train_ds.values()]
    train_datasets = concatenate_datasets(train_datasets)
    if data_args.data_mix == "random":
        train_datasets = train_datasets.shuffle()  
        
    if model_args.reinit:
        logger.info("Reinit lora layer")
        reinit_lora(model, model_args)

    model.print_trainable_parameters()
    logger.info(f"model.modules_to_save: {model.modules_to_save}")
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))


    training_args.remove_unused_columns = False
    
    # Initialize our Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_datasets,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator
    # )
    trainer = RDropTrainer(
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