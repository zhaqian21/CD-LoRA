import torch
import random
from torch.utils.data import DataLoader, Sampler, DataLoader, ConcatDataset
from datasets import Dataset
from functools import partial
from transformers import Trainer
from data_load import *
from torch.nn.utils.rnn import pad_sequence

class CustomDataset(Dataset):
    def __init__(self, datasets, sample_size=2):
        self.datasets = datasets
        self.sample_per_dataset = sample_size
        self.length = len(self.datasets[0])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for dataset in self.datasets:
            # print(type(dataset), dataset)

            # for dataset in self.datasets:
            # # print(type(dataset), dataset)
            #     samples = dataset[idx:idx+self.sample_per_dataset]
            # # print(type(samples), samples)
            # for i in range(self.sample_per_dataset):

            #     batch_input_ids.append(samples['input_ids'][i])

            #     batch_attention_mask.append(samples['attention_mask'][i])

            #     batch_labels.append(samples['labels'][i])
            samples = []
            # length = len(dataset)
            for i in range(self.sample_per_dataset):
                sample_idx = (idx + i) % self.length
                samples.append(dataset[sample_idx])
            
            for sample in samples:
                batch_input_ids.append(sample['input_ids'])
                batch_attention_mask.append(sample['attention_mask'])
                batch_labels.append(sample['labels'])
                # batch_input_ids.append(sample['input_ids'])
                # batch_attention_mask.append(sample['attention_mask'])
                # batch_labels.append(sample['labels'])
        # 将列表转换为tensor
        batch_input_ids = torch.tensor(batch_input_ids)
        batch_attention_mask = torch.tensor(batch_attention_mask)
        batch_labels = torch.tensor(batch_labels)
        
        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'labels': batch_labels
        }

class MultiDatasetSampler(Sampler):
    def __init__(self, datasets, batch_size, ratios):
        self.datasets = datasets
        self.batch_size = batch_size
        self.ratios = ratios
        self.sizes = [len(dataset) for dataset in datasets]
        self.total_size = sum(self.sizes)
        self.num_batches = self.total_size // self.batch_size

        # 计算每个数据集的索引偏移量
        self.offsets = [0] * len(datasets)
        for i in range(1, len(datasets)):
            self.offsets[i] = self.offsets[i - 1] + self.sizes[i - 1]

        # 预先生成所有批次的索引
        self.batch_indices = self._generate_batch_indices()
    def _generate_batch_indices(self):
        batch_indices = []
        for _ in range(self.num_batches):
            indices = []
            for i, dataset in enumerate(self.datasets):
                # 根据比例采样
                n_samples = int(self.batch_size * self.ratios[i])
                dataset_indices = random.sample(range(self.sizes[i]), n_samples)
                # 将数据集索引转换为全局索引
                global_indices = [idx + self.offsets[i] for idx in dataset_indices]
                indices.extend(global_indices)
            batch_indices.extend(indices)
        return batch_indices

    def __iter__(self):
        return iter(self.batch_indices)

    def __len__(self):
        return self.num_batches * self.batch_size

def custom_collate_fn(batch):
    # ... existing code ...
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    # 修改这一行，直接堆叠已经是张量的 labels
    labels = torch.stack([item['labels'] for item in batch])
    
    # # 对 input_ids、attention_mask 和 labels 进行填充
    # input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    # attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    # labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # 使用 -100 填充 labels
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


class CustomDataset2(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset  # 封装 Hugging Face 的 Dataset 对象

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'input_ids': torch.tensor(item['input_ids']),  # 确保转换为张量
            'attention_mask': torch.tensor(item['attention_mask']),  # 确保转换为张量
            'labels': torch.tensor(item['labels'])  # 确保转换为张量
        }


class CustomTrainer_data(Trainer):
    def __init__(self, train_dataloader=None, eval_dataloader=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

    def get_train_dataloader(self):
        if self.train_dataloader is not None:
            return self.train_dataloader
        else:
            return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        if self.eval_dataloader is not None:
            return self.eval_dataloader
        else:
            return super().get_eval_dataloader(eval_dataset)
            
# 自定义Trainer类
class CustomTrainer0(Trainer):
    def __init__(self, dataloader,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_loader = dataloader
        
    def get_train_dataloader(self):
        return self.data_loader
    
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        inputs['labels'] = inputs['labels'].squeeze(0)
        # input_ids = inputs['input_ids']
        # attention_mask = inputs['attention_mask']
        # labels = inputs['labels']
        # print(input_ids.shape, attention_mask.shape, labels.shape)
        outputs = model(**inputs)
        logits_loss = outputs.loss
        hidden_states = outputs.hidden_states
        rep_loss = self.compute_representation_loss(hidden_states)
        
        # print('logit loss: ', logits_loss)
        # print('rep loss: ', representation_loss)
        total_loss = logits_loss + 0.01 * rep_loss
        self.log({"Logit_loss": logits_loss.item(), "Rep_loss": rep_loss.item(), "Total_loss": total_loss.item()})
        return (total_loss, outputs) if return_outputs else total_loss
    
    def pairwise_distance_3d(self, x, y):
        # 检查输入是否为三维矩阵以及特征数是否相同
        if not len(x.shape) == len(y.shape) == 3:
            raise ValueError('Both inputs should be 3D tensors.')
    
        if x.shape[2] != y.shape[2]:
            raise ValueError('The number of features should be the same.')
        
        # 得到 x 和 y 之间的成对距离 形状为 (N, M, length) 的张量
        return torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, 3)

    def gaussian_kernel_matrix(self, x, y, sigmas):
        # 将 sigmas 扩展为形状 (S, 1)
        sigmas = sigmas.view(sigmas.shape[0], 1)
        # print("simgas shape", sigmas.shape)
        # 计算 beta，beta 是高斯核的带宽参数
        beta = 1. / (2. * sigmas)
        # 计算 x 和 y 之间的成对距离 得到形状为 (N, M) 的张量 
        # contigous()保证数据在内存中是连续的 便于之后操作
        dist = self.pairwise_distance_3d(x, y).contiguous()
        # print("dist shape", dist.shape)
        # 将 dist 扩展为形状 (1, N * M * length) 以匹配 beta 的形状 (S, 1)
        dist_ = dist.view(1, -1)
        # print("dist_view shape ", dist_.shape)
        # 计算指数部分 得到形状为 (S, N * M * length) 的张量
        s = torch.matmul(beta, dist_)
        # print("s shape", s.shape)
        # 对第0维度求和 得到形状为 (N * M * length) 的张量
        s_sum = torch.sum(torch.exp(-s), 0)
        # print("s_sum shape", s_sum.shape)
        # print("s_sum", s_sum)
        # 将 s_sum 扩展为形状 (N, M, length) 的张量
        matrix = s_sum.view_as(dist)
        # print("matrix shape", matrix.shape)
        # print("matrix", matrix)
        return matrix
    
    def maximum_mean_discrepancy(self, x, y, kernel= gaussian_kernel_matrix):
        # 计算 x 分布的核矩阵均值
        x_distance = kernel(x, x)
        y_distance = kernel(y, y)
        xy_distance = kernel(x, y)
        # print('xy_distance',xy_distance)
        # print('cost dimention',x_distance.shape, y_distance.shape, xy_distance.shape)
        cost = torch.mean(x_distance)
        # # 计算 y 分布的核矩阵均值
        cost += torch.mean(y_distance)
        # 计算 x 和 y 分布的核矩阵交叉均值
        cost = 2 * torch.mean(xy_distance)
    
        return cost
    
    def mmd_loss(self, source_features, target_features):
        # 定义高斯核参数 sigmas
        # sigmas = [
        #     1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        #     1e3, 1e4, 1e5, 1e6
        # ]
        # # partial创建一个新的函数，这个新函数是基于另一个函数的部分参数已经被固定的版本
        # # Variable创建了一个浮点张量 值是 sigmas 列表中的元素
    
        # gaussian_kernel = partial(
        #     self.gaussian_kernel_matrix, sigmas = Variable(torch.cuda.FloatTensor(sigmas))
        # )
        # print(gaussian_kernel)
        sigmas = torch.tensor(
            [
                1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
                1e3, 1e4, 1e5, 1e6
            ],
            dtype=torch.bfloat16,  # 设置数据类型
            device=source_features.device  # 将 sigmas 移动到与输入特征相同的设备
        )

        # 创建新的高斯核函数，基于 sigmas
        gaussian_kernel = partial(self.gaussian_kernel_matrix, sigmas=sigmas)
        loss_value = self.maximum_mean_discrepancy(source_features, target_features, kernel= gaussian_kernel)
        loss_value = loss_value
    
        return loss_value
    

    def compute_representation_loss(self, hidden_states):
        last_5_layers = hidden_states[-5:]
        rep_loss = 0
        for layer in last_5_layers:
            # print('layer:', layer.requires_grad)
            # 将批次平均分为5份
            # print(layer.shape)
            num_chunks = 5
            split_hidden_states = torch.chunk(layer, num_chunks, dim=0)
            # print(split_hidden_states[0].shape, len(split_hidden_states))
            mean_hidden_state = torch.mean(torch.cat(split_hidden_states, dim=0), dim=0, keepdim=True)
            # print(mean_hidden_state.shape)
        
            layer_loss = 0
            for hidden_state in split_hidden_states:
                # print(hidden_state.shape)
                state_loss = self.mmd_loss(hidden_state, mean_hidden_state)
                layer_loss += state_loss
            layer_loss /= len(split_hidden_states)
            # print('Layer loss:', layer_loss)
            rep_loss += layer_loss
        rep_loss /= len(last_5_layers)
        return rep_loss


# 自定义Trainer类
class CustomTrainer(Trainer):
    def __init__(self, dataloader,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_loader = dataloader
        self.kernel_num = 19
        self.kernel_mul = 2.0
        self.bandwidth_list = None
        
    def get_train_dataloader(self):
        return self.data_loader
    
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        inputs['labels'] = inputs['labels'].squeeze(0)
        # input_ids = inputs['input_ids']
        # attention_mask = inputs['attention_mask']
        # labels = inputs['labels']
        # print(input_ids.shape, attention_mask.shape, labels.shape)
        outputs = model(**inputs)
        logits_loss = outputs.loss
        hidden_states = outputs.hidden_states
        rep_loss = self.compute_representation_loss(hidden_states)
        
        # print('logit loss: ', logits_loss)
        # print('rep loss: ', representation_loss)
        total_loss = logits_loss + 0.01 * rep_loss
        self.log({"Logit_loss": logits_loss.item(), "Rep_loss": rep_loss.item(), "Total_loss": total_loss.item()})
        return (total_loss, outputs) if return_outputs else total_loss
    
    def compute_bandwidth(self, source, target):
        total = torch.cat([source, target], dim=0)
        total = total.float()
        L2_distance = torch.cdist(total.view(total.size(0), -1), total.view(total.size(0), -1), p=2)
        L2_distance = L2_distance.to(source.dtype)
        bandwidth = torch.sum(L2_distance.data) / (source.size(0)**2 - source.size(0))
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        return torch.tensor([bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)], device=source.device)
    
    def guassian_kernel(self, x, y, bandwidth_list):
        # x = x.expand(y.size(0), -1, -1)  # 将 x 扩展为 (y.size(0), feature_dim, length)
        # 计算 L2 距离，适用于 (batch_size, feature_dim, length) 的数据
        dist_sq = torch.sum((x - y)**2, dim=(1, 2)).unsqueeze(1)  # 在 feature_dim 和 length 维度上求和
        # 计算多尺度高斯核
        kernels = torch.exp(-dist_sq / (2 * torch.tensor(bandwidth_list).unsqueeze(0).to(x.device)**2))
        return kernels.sum(dim=1)

    def mmd_loss(self, source, target):
        n_s = source.shape[0]
        
        # 如果 target 只有一个样本，则重复它
        if target.shape[0] == 1:
            target = target.expand(n_s, -1, -1)  # 将 target 扩展为 (n_s, feature_dim, length)
        assert n_s % 2 == 0, "n_s must be even"
        # half_n = n_s // 2
        
        # 将源数据和目标数据配对
        x1 = source[0::2]  # 奇数索引
        x2 = source[1::2]  # 偶数索引
        y1 = target[0::2]  # 奇数索引
        y2 = target[1::2]  # 偶数索引
        # print(x1.shape, x2.shape, y1.shape, y2.shape)
        
        if self.bandwidth_list is None:
            self.bandwidth_list = self.compute_bandwidth(source, target)

        k_xx = self.guassian_kernel(x1, x2, self.bandwidth_list)
        k_yy = self.guassian_kernel(y1, y2, self.bandwidth_list)
        k_xy1 = self.guassian_kernel(x1, y2, self.bandwidth_list)
        k_xy2 = self.guassian_kernel(x2, y1, self.bandwidth_list)

        g = k_xx + k_yy - k_xy1 - k_xy2
        return torch.mean(g)
        
    def compute_representation_loss(self, hidden_states):
        last_5_layers = hidden_states[-5:]
        rep_loss = 0
        for layer in last_5_layers:
            # print('layer:', layer.requires_grad)
            # 将批次平均分为5份
            # print(layer.shape)
            num_chunks = 5
            split_hidden_states = torch.chunk(layer, num_chunks, dim=0)
            # print(split_hidden_states[0].shape, len(split_hidden_states))
            mean_hidden_state = torch.mean(torch.cat(split_hidden_states, dim=0), dim=0, keepdim=True)
            # print(mean_hidden_state.shape)
        
            layer_loss = 0
            state_losses = torch.stack([self.mmd_loss(hidden_state, mean_hidden_state) for hidden_state in split_hidden_states])
            layer_loss = torch.mean(state_losses)
            # print('Layer loss:', layer_loss)
            rep_loss += layer_loss
        rep_loss /= len(last_5_layers)
        return rep_loss
    
class CustomTrainer2(Trainer):
    def __init__(self, dataloader,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_loader = dataloader
        self.kernel_num = 19
        self.kernel_mul = 2.0
        self.bandwidth_list = None
        
    def get_train_dataloader(self):
        return self.data_loader
    
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        inputs['labels'] = inputs['labels'].squeeze(0)
        # input_ids = inputs['input_ids']
        # attention_mask = inputs['attention_mask']
        # labels = inputs['labels']
        # print(input_ids.shape, attention_mask.shape, labels.shape)
        outputs = model(**inputs)
        logits_loss = outputs.loss
        hidden_states = outputs.hidden_states
        rep_loss = self.compute_representation_loss(hidden_states)
        
        # print('logit loss: ', logits_loss)
        # print('rep loss: ', representation_loss)
        total_loss = logits_loss + 0.01 * rep_loss
        self.log({"Logit_loss": logits_loss.item(), "Rep_loss": rep_loss.item(), "Total_loss": total_loss.item()})
        return (total_loss, outputs) if return_outputs else total_loss
    
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        inputs['labels'] = inputs['labels'].squeeze(0)
        # input_ids = inputs['input_ids']
        # attention_mask = inputs['attention_mask']
        # labels = inputs['labels']
        # print(input_ids.shape, attention_mask.shape, labels.shape)
        outputs = model(**inputs)
        logits_loss = outputs.loss
        hidden_states = outputs.hidden_states
        rep_loss = self.compute_representation_loss(hidden_states)
        
        # print('logit loss: ', logits_loss)
        # print('rep loss: ', representation_loss)
        total_loss = logits_loss + 0.01 * rep_loss
        self.log({"Logit_loss": logits_loss.item(), "Rep_loss": rep_loss.item(), "Total_loss": total_loss.item()})
        return (total_loss, outputs) if return_outputs else total_loss
    
    def guassian_kernel(self, source, target):
        # 将源数据和目标数据拼接
        total = torch.cat([source, target], dim=0)
        total_flat = total.view(total.size(0), -1)  # 展平为 (n_samples, feature * length)
        
        # 计算 L2 距离
        x_norm = torch.sum(total_flat ** 2, dim=1, keepdim=True)
        dist_sq = x_norm - 2 * torch.matmul(total_flat, total_flat.t()) + x_norm.t()
        dist_sq = torch.clamp(dist_sq, min=0.0)  # 确保距离非负
        
        # 如果带宽列表未计算，则计算带宽列表
        # if self.bandwidth_list is None:
        n_samples = total.size(0)
        bandwidth = torch.sum(dist_sq) / (n_samples**2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        self.bandwidth_list = bandwidth * (self.kernel_mul ** torch.arange(self.kernel_num, device=source.device))
        
        # 计算多尺度高斯核
        bandwidth_sq = 2 * (self.bandwidth_list ** 2).unsqueeze(0).unsqueeze(0).to(source.device)
        exponent = -dist_sq.unsqueeze(2) / bandwidth_sq
        kernels = torch.exp(exponent)
        return torch.sum(kernels, dim=2)

    def mmd_loss(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target)
        XX = torch.mean(kernels[:batch_size, :batch_size])
        YY = torch.mean(kernels[batch_size:, batch_size:])
        XY = torch.mean(kernels[:batch_size, batch_size:])
        YX = torch.mean(kernels[batch_size:, :batch_size])
        loss = torch.mean(XX + YY - XY - YX)
        return loss
        
    def compute_representation_loss(self, hidden_states):
        last_5_layers = hidden_states[-5:]
        rep_loss = 0
        for layer in last_5_layers:
            # print('layer:', layer.requires_grad)
            # 将批次平均分为5份
            # print(layer.shape)
            num_chunks = 5
            split_hidden_states = torch.chunk(layer, num_chunks, dim=0)
            # print(split_hidden_states[0].shape, len(split_hidden_states))
            mean_hidden_state = torch.mean(torch.cat(split_hidden_states, dim=0), dim=0, keepdim=True)
            # print(mean_hidden_state.shape)
        
            layer_loss = 0
            state_losses = torch.stack([self.mmd_loss(hidden_state, mean_hidden_state) for hidden_state in split_hidden_states])
            layer_loss = torch.mean(state_losses)
            # print('Layer loss:', layer_loss)
            rep_loss += layer_loss
        rep_loss /= len(last_5_layers)
        return rep_loss

class CustomTrainer3(Trainer):
    def __init__(self, dataloader,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_loader = dataloader
        self.kernel_num = 19
        self.kernel_mul = 2.0
        self.bandwidth_list = None
        
    def get_train_dataloader(self):
        return self.data_loader
    
    def guassian_kernel(self, source, target):
        # 计算源数据和目标数据的总样本数
        n_samples = int(source.size()[0]) + int(target.size()[0])
        
        # 将源数据和目标数据在第0维上拼接
        total = torch.cat([source, target], dim=0)
        
        # 扩展 total 以便计算两两之间的距离
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2)))
        
        # 计算 L2 距离
        L2_distance = ((total0-total1)**2).sum(3).sum(2)
        
        if self.bandwidth_list is None:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
            
            # 调整带宽
            bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
            
            # 生成带宽列表
            self.bandwidth_list = [bandwidth * (self.kernel_mul**i)
                            for i in range(self.kernel_num)]
        
        # 计算每个带宽对应的核值
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                    for bandwidth_temp in self.bandwidth_list]
        
        # 返回所有核值的和
        return sum(kernel_val)

    def mmd_loss(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target)
        XX = torch.mean(kernels[:batch_size, :batch_size])
        YY = torch.mean(kernels[batch_size:, batch_size:])
        XY = torch.mean(kernels[:batch_size, batch_size:])
        YX = torch.mean(kernels[batch_size:, :batch_size])
        loss = torch.mean(XX + YY - XY - YX)
        return loss
        
    def compute_representation_loss(self, hidden_states):
        last_5_layers = hidden_states[-5:]
        rep_loss = 0
        for layer in last_5_layers:
            # print('layer:', layer.requires_grad)
            # 将批次平均分为5份
            # print(layer.shape)
            num_chunks = 5
            split_hidden_states = torch.chunk(layer, num_chunks, dim=0)
            # print(split_hidden_states[0].shape, len(split_hidden_states))
            mean_hidden_state = torch.mean(torch.cat(split_hidden_states, dim=0), dim=0, keepdim=True)
            # print(mean_hidden_state.shape)
        
            layer_loss = 0
            state_losses = torch.stack([self.mmd_loss(hidden_state, mean_hidden_state) for hidden_state in split_hidden_states])
            layer_loss = torch.mean(state_losses)
            # print('Layer loss:', layer_loss)
            rep_loss += layer_loss
        rep_loss /= len(last_5_layers)
        return rep_loss

class CustomTrainer4(Trainer):
    def __init__(self, train_dataloader=None, eval_dataloader=None, *args, **kwargs):
        self.custom_train_dataloader = kwargs.pop('custom_train_dataloader', None)
        super().__init__(*args, **kwargs)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

    def get_train_dataloader(self):
        # 如果提供了自定义的 DataLoader，则直接返回
        if self.train_dataloader is not None:
            return self.train_dataloader
        # 否则使用默认的逻辑
        return super().get_train_dataloader()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        # inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        # inputs['labels'] = inputs['labels'].squeeze(0)
        # input_ids = inputs['input_ids']
        # attention_mask = inputs['attention_mask']
        # labels = inputs['labels']
        # print(input_ids.shape, attention_mask.shape, labels.shape)
        outputs = model(**inputs)
        logits_loss = outputs.loss
        hidden_states = outputs.hidden_states
        rep_loss = self.compute_representation_loss(hidden_states)
        
        # print('logit loss: ', logits_loss)
        # print('rep loss: ', representation_loss)
        total_loss = logits_loss + 0.05 * rep_loss
        self.log({"Logit_loss": logits_loss.item(), "Rep_loss": rep_loss.item(), "Total_loss": total_loss.item()})
        return (total_loss, outputs) if return_outputs else total_loss
    
    def pairwise_distance_3d(self, x, y):
        # 检查输入是否为三维矩阵以及特征数是否相同
        if not len(x.shape) == len(y.shape) == 3:
            raise ValueError('Both inputs should be 3D tensors.')
    
        if x.shape[2] != y.shape[2]:
            raise ValueError('The number of features should be the same.')
        
        # 得到 x 和 y 之间的成对距离 形状为 (N, M, length) 的张量
        return torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, 3)

    def gaussian_kernel_matrix(self, x, y, sigmas):
        # 将 sigmas 扩展为形状 (S, 1)
        sigmas = sigmas.view(sigmas.shape[0], 1)
        # print("simgas shape", sigmas.shape)
        # 计算 beta，beta 是高斯核的带宽参数
        beta = 1. / (2. * sigmas)
        # 计算 x 和 y 之间的成对距离 得到形状为 (N, M) 的张量 
        # contigous()保证数据在内存中是连续的 便于之后操作
        dist = self.pairwise_distance_3d(x, y).contiguous()
        # print("dist shape", dist.shape)
        # 将 dist 扩展为形状 (1, N * M * length) 以匹配 beta 的形状 (S, 1)
        dist_ = dist.view(1, -1)
        # print("dist_view shape ", dist_.shape)
        # 计算指数部分 得到形状为 (S, N * M * length) 的张量
        s = torch.matmul(beta, dist_)
        # print("s shape", s.shape)
        # 对第0维度求和 得到形状为 (N * M * length) 的张量
        s_sum = torch.sum(torch.exp(-s), 0)
        # print("s_sum shape", s_sum.shape)
        # print("s_sum", s_sum)
        # 将 s_sum 扩展为形状 (N, M, length) 的张量
        matrix = s_sum.view_as(dist)
        # print("matrix shape", matrix.shape)
        # print("matrix", matrix)
        return matrix
    
    def maximum_mean_discrepancy(self, x, y, kernel= gaussian_kernel_matrix):
        # 计算 x 分布的核矩阵均值
        x_distance = kernel(x, x)
        y_distance = kernel(y, y)
        xy_distance = kernel(x, y)
        # print('xy_distance',xy_distance)
        # print('cost dimention',x_distance.shape, y_distance.shape, xy_distance.shape)
        cost = torch.mean(x_distance)
        # # 计算 y 分布的核矩阵均值
        cost += torch.mean(y_distance)
        # 计算 x 和 y 分布的核矩阵交叉均值
        cost = 2 * torch.mean(xy_distance)
    
        return cost
    
    def mmd_loss(self, source_features, target_features):
        # 定义高斯核参数 sigmas
        # sigmas = [
        #     1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        #     1e3, 1e4, 1e5, 1e6
        # ]
        # # partial创建一个新的函数，这个新函数是基于另一个函数的部分参数已经被固定的版本
        # # Variable创建了一个浮点张量 值是 sigmas 列表中的元素
    
        # gaussian_kernel = partial(
        #     self.gaussian_kernel_matrix, sigmas = Variable(torch.cuda.FloatTensor(sigmas))
        # )
        # print(gaussian_kernel)
        sigmas = torch.tensor(
            [
                1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
                1e3, 1e4, 1e5, 1e6
            ],
            dtype=torch.bfloat16,  # 设置数据类型
            device=source_features.device  # 将 sigmas 移动到与输入特征相同的设备
        )

        # 创建新的高斯核函数，基于 sigmas
        gaussian_kernel = partial(self.gaussian_kernel_matrix, sigmas=sigmas)
        loss_value = self.maximum_mean_discrepancy(source_features, target_features, kernel= gaussian_kernel)
        loss_value = loss_value
    
        return loss_value
    

    def compute_representation_loss(self, hidden_states):
        last_5_layers = hidden_states[-5:]
        rep_loss = 0
        for layer in last_5_layers:
            # print('layer:', layer.requires_grad)
            # 将批次平均分为5份
            # print(layer.shape)
            num_chunks = 5
            split_hidden_states = torch.chunk(layer, num_chunks, dim=0)
            # print(split_hidden_states[0].shape, len(split_hidden_states))
            mean_hidden_state = torch.mean(torch.cat(split_hidden_states, dim=0), dim=0, keepdim=True)
            # print(mean_hidden_state.shape)
        
            layer_loss = 0
            for hidden_state in split_hidden_states:
                # print(hidden_state.shape)
                state_loss = self.mmd_loss(hidden_state, mean_hidden_state)
                layer_loss += state_loss
            layer_loss /= len(split_hidden_states)
            # print('Layer loss:', layer_loss)
            rep_loss += layer_loss
        rep_loss /= len(last_5_layers)
        return rep_loss
    