import torch
from itertools import combinations
# def compute_representation_loss(features):
#     num_layers = 0
#     print("features", features.shape)
#     steps = 20  # 假设 self.grad_step 是梯度累计步数
#     step_chunks = torch.chunk(features, steps, dim=0)  # 按梯度累计步数分割
#     print("step_chunks", len(step_chunks))
#     # 按任务分割每个步的特征，并将各步内同任务的特征合并
#     groups = [torch.cat([torch.chunk(step, 5, dim=0)[task_idx] for step in step_chunks], dim=0)
#             for task_idx in range(5)]
#     print(groups[0].shape, len(groups))
#     # groups = torch.chunk(features, self.num_tasks, dim=0)
#     # layer_total = torch.tensor(0.0, device=self.model.device)


# input_tensor = torch.randn(95, 512, 8)  # Assuming CUDA is available
# compute_representation_loss(input_tensor)

pi1 = torch.tensor([0.1, 0.2, 0.7])
pi2 = torch.tensor([0.7, 0.2, 0.1])

P = torch.tensor([[0.65, 0.28, 0.07],[0.15,0.67,0.18],[0.12,0.36,0.52]])

for i in range(10):
    print(i)
    P = P @ P
    result1 = pi1 @ P
    print("result1", result1)
    # result2 = pi2 @ P

    result2 = pi2 @ P
    print("result2", result2)