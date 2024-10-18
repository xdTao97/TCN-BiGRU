import torch
import torch.nn as nn

from model import TCNSENetBiGRUGlobalAttention


# 定义模型参数
batch_size = 64
input_dim = 7  # 输入的特征维度
output_dim = 1  # 输出的特征维度
num_channels = [32, 64]  # 每个TemporalBlock中的输出通道数
kernel_size = 3  # 卷积核大小
dropout = 0.5  # Dropout概率
# BiGRU 层数和维度数
hidden_layer_sizes = [32, 64]
# 全局注意力维度数
attention_dim = hidden_layer_sizes[-1]  # 注意力层维度 默认为 BiGRU输出层维度

# 输入数据维度为[batch_size, sequence_length, input_dim]
# 输出维度为[batch_size, output_dim]
model = TCNSENetBiGRUGlobalAttention(batch_size, input_dim, output_dim, num_channels, kernel_size, hidden_layer_sizes, attention_dim, dropout)

# 定义损失函数和优化函数
loss_function = nn.MSELoss(reduction='sum')  # loss
learn_rate = 0.0003
optimizer = torch.optim.Adam(model.parameters(), learn_rate)  # 优化器

# 看下这个网络结构总共有多少个参数
def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')

count_parameters(model)

print(model)
