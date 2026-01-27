# 定义种子列表
seeds=(27)

# 定义基础输出目录和种子记录文件
base_output_dir="./output/Q-B-3_mt_lora"


# 遍历种子列表并运行训练脚本
for seed in "${seeds[@]}"; do
    # 定义与种子相关的输出目录
    output_dir="${base_output_dir}_seed_${seed}"
    
    echo "Running experiment with seed: $seed"
    echo "Output directory: $output_dir"
    
    # 创建输出目录
    mkdir -p "$output_dir"
    
    # 运行 Python 脚本
    python train_bs.py \
        --seed ${seed} \
        --output_dir "$output_dir"
    
    echo "Experiment with seed $seed completed."
done

echo "All experiments completed. Seed values recorded in $output_file."