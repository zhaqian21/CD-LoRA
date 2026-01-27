import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.manifold import TSNE

# ======================
# 基本配置
# ======================
model_path = "/root/autodl-tmp/qwen/Qwen2.5-14B"
# lora_path = "/root/autodl-tmp/HydraLoRA/HydraLoRA/output/drop2_7B-rlora_unit_seed7_rdrop_alpha0.0_12tasks/checkpoint-491"
# lora_path = "/root/autodl-tmp/HydraLoRA/HydraLoRA/output/gauss_drop2_7B-rlora_unit_seed7_rdrop_alpha0.05_12tasks/checkpoint-491"
# lora_path = "/root/autodl-tmp/HydraLoRA/HydraLoRA/output/***gauss0.0_drop2_7B-rlora_unit_seed7_12tasks/checkpoint-491"
lora_path = "/root/autodl-tmp/HydraLoRA/HydraLoRA/output/***gauss0.05_drop2_7B-rlora_unit_seed7_12tasks/checkpoint-491"
exp_name = os.path.basename(os.path.dirname(lora_path))

SAVE_DIR = "./A_analysis"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_LAYERS = 48
MODULES = ["gate_proj", "down_proj"]

# 每个任务采样多少个样本
# MAX_SAMPLES_PER_TASK = 80
MAX_SAMPLES_PER_TASK = 10

# ======================
# 数据集加载函数（与你 ft_rlora.py 相同）
# ======================
def load_tokenized_datasets(base_dir, dataset_names):
    out = {}
    for name in dataset_names:
        path = os.path.join(base_dir, name)
        ds = load_from_disk(path)
        out[name] = ds
    return out


# ----------- 加载任务数据 -------------
glue_token_dir = "/root/autodl-tmp/data/glue_tokenized"
glue_train_ds = load_tokenized_datasets(glue_token_dir, ["qnli_train", "cola_train", "mnli_train", "qqp_train", "sst2_train"])

cs_token_dir = "/root/autodl-tmp/data/commonsense_tokenized"
cs_train_ds = load_tokenized_datasets(cs_token_dir, ["piqa_train", "winogrande_train", "siqa_train", "boolq_train", "hellaswag_train"])

arc_train_ds = load_from_disk("/root/autodl-tmp/data/arc_tokenized/arc_train")
arc_test_ds = load_from_disk("/root/autodl-tmp/data/arc_tokenized/arc_test")
arc_ds = {"arc": concatenate_datasets([arc_train_ds, arc_test_ds])}

gsm_train_ds = load_from_disk("/root/autodl-tmp/data/gsm8k_tokenized/gsm8k_train")
gsm_train_ds = {"gsm8k": gsm_train_ds}

train_ds = {**glue_train_ds, **cs_train_ds, **arc_ds, **gsm_train_ds}
task_names = list(train_ds.keys())



# ======================
# 载入模型
# ======================
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading Base Model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map={"": 0}, torch_dtype=torch.float32
)

print("Loading LoRA...")
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

print("Model loaded successfully.\n")


# ======================
# Hook + 特征缓存
# ======================
def register_loraA_hook(model, layer_id, module_name, storage_list):
    """
    storage_list: 外部传入的 list，用于存储 features
    """

    # 定位 LoRA A 层
    loraA_module = eval(
        f"model.base_model.model.model.layers[{layer_id}].mlp.{module_name}.lora_A"
    )

    def hook_fn(m, inp, out):
        # inp[0]: (batch, seq_len, hidden)
        x = inp[0]                   # 输入 x
        x = x.mean(dim=1)            # 减掉 seq_len
        x = x.squeeze(0)             # 去掉 batch 维度
        x = x.flatten().detach().float().cpu().numpy()   # 强制变成 1D 向量
        storage_list.append((x, layer_id))

    loraA_module.register_forward_hook(hook_fn)


# ======================
# 主流程函数：处理一个 module (gate_proj / up_proj / down_proj)
# ======================
def extract_features_for_module(module_name):
    print(f"\n==== Processing module: {module_name} ====\n")

    features = []
    labels = []

    # 在所有层注册 hook
    storage = []
    for layer in range(NUM_LAYERS):
        register_loraA_hook(model, layer, module_name, storage)

    # 对任务循环
    for task in task_names:
        ds = train_ds[task]
        print(f"Task {task}: {len(ds)} samples")

        count = 0
        for example in ds:
            if count >= MAX_SAMPLES_PER_TASK:
                break

            ids = torch.tensor([example["input_ids"]], device="cuda")

            # forward → 激活 hooks
            with torch.no_grad():
                model(ids)

            # 从 storage 收集 28 层对应的特征
            for feat, layer_id in storage:
                features.append(feat)
                labels.append((task, layer_id))

            storage.clear()
            count += 1

    print(f"Total collected points: {len(features)}")
    return np.array(features), labels


# ======================
# t-SNE 绘图
# ======================
def plot_tsne(features, labels, module_name):

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    emb2d = tsne.fit_transform(features)

    print("Plotting...")

    # 设置颜色
    unique_tasks = task_names
    cmap = plt.get_cmap("tab20")
    task2color = {task: cmap(i%20) for i, task in enumerate(unique_tasks)}

    # 点大小（层数映射到 10~120）
    def layer_to_size(layer_id):
        return 10 + (layer_id / (NUM_LAYERS - 1)) * 110

    plt.figure(figsize=(10, 8))
    for i, (task, layer_id) in enumerate(labels):
        plt.scatter(
            emb2d[i, 0],
            emb2d[i, 1],
            color=task2color[task],
            s=layer_to_size(layer_id),
            alpha=0.65,
        )

    plt.title(f"t-SNE of LoRA A ({module_name}) across layers & tasks", fontsize=14)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")

    # legend
    for task in unique_tasks:
        plt.scatter([], [], color=task2color[task], label=task)
    plt.legend(loc="best")
    out_path = os.path.join(SAVE_DIR, f"{module_name}_tsne_{exp_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}\n")


# ======================
# 主执行逻辑：三张图
# ======================
if __name__ == "__main__":
    for module in MODULES:
        feats, lbls = extract_features_for_module(module)
        plot_tsne(feats, lbls, module)

    print("\nAll visualizations saved to ./analysis/")
