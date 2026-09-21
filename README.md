# CD-LoRA

Code for multi-task and task-aware LoRA fine-tuning experiments.

## Repository structure

- `fine-tuning.py`: instruction fine-tuning with Hugging Face Transformers and PEFT.
- `fine-tuning.sh`: example training configuration.
- `ft_cs.py`, `ft_mt.py`, `ft_new.py`, `ft_rlora.py`: additional LoRA and multi-task experiments.
- `build_dataset.py`, `data_load.py`, `data_process.py`, `dataloader.py`, `load_data.py`, `load_math.py`: data loading, processing, and sampling utilities.
- `test_glue.py`, `test_cs.py`, `test_fc.py`, `eval_bbh.py`: evaluation scripts.
- `analysis_*.py`, `ana_cos.py`, `parameter.py`, `reinit.py`, `remove_tasktype.py`: parameter and representation analysis utilities.
- `custom.py`: custom datasets, samplers, trainers, and representation-alignment losses.

## Requirements

The project uses Python, PyTorch, Transformers, Datasets, PEFT, and Accelerate:

```bash
pip install torch transformers datasets peft accelerate sentencepiece tqdm numpy
```

## Minimal training command

`fine-tuning.py` uses `HfArgumentParser` for model, dataset, and training arguments. Replace the placeholder paths with local paths:

```bash
python fine-tuning.py \
  --model_name_or_path /path/to/base-model \
  --tokenizer_name_or_path /path/to/base-model \
  --dataset_dir /path/to/train-data \
  --validation_file /path/to/validation.json \
  --output_dir ./output \
  --do_train \
  --do_eval \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --max_seq_length 512 \
  --trainable q_proj,v_proj \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.1
```

Update the model, dataset, cache, and output paths for your environment. The supplied examples assume CUDA.

## Paper

[CD-LoRA: Consistency-Driven Low-Rank Adaptation for Multi-Task Fine-Tuning](https://arxiv.org/abs/2608.21909)
