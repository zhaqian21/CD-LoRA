export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

# User-defined parameters
lr=0.0002
lora_rank=4
lora_alpha=32
lora_trainable="gate_proj,down_proj,up_proj"
lora_dropout=0.1                 
pretrained_model="/root/autodl-tmp/qwen/Qwen2.5-14B"
tokenizer_path="/root/autodl-tmp/qwen/Qwen2.5-14B"
# dataset_dir="/root/autodl-tmp/data/gsm8k"
# validation_file="/root/autodl-tmp/data/example_data/test.json"
dataset_dir="data/processed_data"
validation_file="data/processed_data/test.json"
# per_device_train_batch_size=2
per_device_train_batch_size=3
per_device_eval_batch_size=16
gradient_accumulation_steps=64
max_seq_length=512
seed=7
output_dir="/root/autodl-tmp/HydraLoRA/HydraLoRA/output"
tokenized_dir="/root/autodl-tmp/data/gsm8k_tokenized"
processed_dir="/root/autodl-tmp/data/gsm8k_processed"
# lora_b_nums=3  # Developer-specific, k-means, or DBSCAN et al.
lora_b_nums=3

mix="no"
init_mode="simple"
torch_dtype="bfloat16"
lora_A="unit"
lora_B="unit"
init_scale="stable"
stable_gamma=64
init_bs=5
sample_size=8000
# sample_size=2000
# exp_name="Q-B-3-rlora-mt5_${lora_A}_${mix}_seed${seed}_head${lora_b_nums}"
# exp_name="7B_xnodrop_noroute_rlora-mt5_${lora_A}_${mix}_seed${seed}_drop${lora_dropout}_rdrop_alpha${rdrop_alpha}_flanv2"

# seeds=(7)
mixes=("no")
lora_dropouts=(0.2)
rdrop_alphas=(0.05)
for rdrop_alpha in "${rdrop_alphas[@]}"; do
  for lora_dropout in "${lora_dropouts[@]}"; do
    exp_name="rlora_14B_gauss${rdrop_alpha}_7B-rlora_${lora_A}_seed${seed}_12tasks"
 
    # --evaluation_strategy epoch \
    # --do_predict \

        CUDA_VISIBLE_DEVICES=0 \
        python   ft_rlora.py \
            --use_router True \
            --rdrop_alpha ${rdrop_alpha} \
            --model_name_or_path ${pretrained_model} \
            --tokenizer_name_or_path ${tokenizer_path} \
            --dataset_dir ${dataset_dir} \
            --processed_dir ${processed_dir} \
            --tokenized_dir ${tokenized_dir} \
            --per_device_train_batch_size ${per_device_train_batch_size} \
            --per_device_eval_batch_size ${per_device_eval_batch_size} \
            --reinit True \
            --init_scale ${init_scale} \
            --rank_stablization False \
            --sample True \
            --sample_size ${sample_size} \
            --data_mix ${mix} \
            --init_mode ${init_mode} \
            --lora_A_init ${lora_A} \
            --lora_B_init ${lora_B} \
            --stable_gamma ${stable_gamma} \
            --init_bs ${init_bs} \
            --do_train \
            --do_eval \
            --seed ${seed} \
            --bf16 True\
            --num_train_epochs 1 \
            --lr_scheduler_type cosine \
            --learning_rate ${lr} \
            --warmup_ratio 0.03 \
            --weight_decay 0 \
            --logging_strategy steps \
            --logging_steps 10 \
            --save_strategy steps \
            --save_total_limit 5 \
            --save_steps 1500 \
            --gradient_accumulation_steps ${gradient_accumulation_steps} \
            --max_seq_length ${max_seq_length} \
            --output_dir ${output_dir}/${exp_name} \
            --logging_first_step True \
            --lora_rank ${lora_rank} \
            --lora_alpha ${lora_alpha} \
            --lora_nums ${lora_b_nums} \
            --trainable ${lora_trainable} \
            --lora_dropout ${lora_dropout} \
            --torch_dtype ${torch_dtype} \
            --validation_file ${validation_file} \
            --load_in_kbits 16 \
            --overwrite_output_dir
  done
done